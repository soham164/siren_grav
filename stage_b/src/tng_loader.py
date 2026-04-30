"""
src/tng_loader.py  —  IllustrisTNG Data Access
===============================================
Handles everything between "I have an API key" and "I have clean numpy
arrays of particle positions, velocities, and masses for one halo."

Two modes of operation:

MODE 1 — API download (requires internet + API key):
    Downloads halo data directly from the TNG REST API.
    Each halo is a few hundred MB — manageable without downloading
    the full 500 GB snapshot.

MODE 2 — Local HDF5 (if you have downloaded a snapshot):
    Reads directly from local HDF5 files using h5py.
    Faster, works offline.

IllustrisTNG data model:
    - The simulation box contains ~18,000 "FoF groups" (halos)
    - Each group is identified by a GroupID (0 = most massive)
    - Within each group, subhalos are identified by SubhaloID
    - Subhalo 0 within each group is the central (most massive) subhalo
    - We always use the central subhalo for the potential field

Particle types in TNG:
    PartType0 : gas
    PartType1 : dark matter  ← primary target for this research
    PartType4 : stars
    PartType5 : black holes  (ignored — too few)
"""

import numpy as np
import os
import json
import time

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


# ── TNG API configuration ─────────────────────────────────────────────────────
TNG_BASE_URL  = "https://www.tng-project.org/api"
TNG_SIM       = "TNG100-1"
TNG_SNAPSHOT  = 99          # z=0


class TNGDownloader:
    """
    Downloads halo data from the IllustrisTNG public REST API.

    Usage:
        dl = TNGDownloader(api_key="your_key_here")
        halos = dl.get_top_halos(n=5)
        data  = dl.download_halo_particles(halos[0], output_dir="data/")

    Getting an API key:
        1. Register at https://www.tng-project.org/users/register/
        2. Go to your profile → API key
        3. Free for academic use

    Parameters
    ----------
    api_key    : str  — your TNG API key
    simulation : str  — "TNG100-1" (default), "TNG50-1", etc.
    snapshot   : int  — 99 for z=0
    """

    def __init__(self, api_key, simulation=TNG_SIM, snapshot=TNG_SNAPSHOT):
        if not REQUESTS_AVAILABLE:
            raise ImportError("pip install requests")

        self.api_key    = api_key
        self.simulation = simulation
        self.snapshot   = snapshot
        self.base_url   = f"{TNG_BASE_URL}/{simulation}/snapshots/{snapshot}"
        self.headers    = {"api-key": api_key}

        # Verify connection
        resp = self._get(TNG_BASE_URL)
        print(f"TNG API connected. Available simulations: "
              f"{[s['name'] for s in resp.get('simulations', [])][:5]}")

    def _get(self, url, params=None, stream=False):
        """Make a GET request with retry logic."""
        for attempt in range(3):
            try:
                r = requests.get(url, headers=self.headers,
                                 params=params, stream=stream, timeout=30)
                r.raise_for_status()
                if stream:
                    return r
                
                # Debug: print response if not JSON
                try:
                    return r.json()
                except json.JSONDecodeError:
                    print(f"  API returned non-JSON response:")
                    print(f"  Status: {r.status_code}")
                    print(f"  Content preview: {r.text[:500]}")
                    raise
                    
            except Exception as e:
                if attempt < 2:
                    print(f"  Retry {attempt+1}/3: {e}")
                    time.sleep(2 ** attempt)
                else:
                    raise

    def get_top_halos(self, n=5, min_dm_particles=10_000):
        """
        Get the N most massive halos from the FoF catalogue.

        Filters:
            - Minimum dark matter particle count (ensures well-resolved halos)
            - Only central subhalos (SubhaloID == 0 within group)

        Returns list of dicts with halo metadata.
        """
        print(f"\nFetching top {n} halos from {self.simulation} Snapshot {self.snapshot}...")

        # Correct TNG API endpoint for groups (FoF halos)
        url    = f"{TNG_BASE_URL}/{self.simulation}/snapshots/{self.snapshot}/halos/"
        params = {
            "limit"       : n * 3,   # fetch extra in case some fail filter
            "order_by"    : "-Group_M_Crit200",   # largest first
        }
        print("DEBUG URL:", url)
        result = self._get(url, params=params)
        halos  = result.get("results", [])

        # Filter and enrich
        selected = []
        for halo in halos:
            gid = halo["id"]
            # Get detailed halo info
            detail = self._get(halo["url"])
            n_dm   = detail.get("Group_Nsubs", 0)

            selected.append({
                "group_id"   : gid,
                "url"        : halo["url"],
                "M200"       : detail.get("Group_M_Crit200", 0),   # 1e10 M_sun/h
                "R200"       : detail.get("Group_R_Crit200", 0),   # ckpc/h
                "pos_x"      : detail.get("GroupPos_x", 0),        # ckpc/h
                "pos_y"      : detail.get("GroupPos_y", 0),
                "pos_z"      : detail.get("GroupPos_z", 0),
                "n_subhalos" : n_dm,
            })

            if len(selected) >= n:
                break

        print(f"Selected {len(selected)} halos:")
        for i, h in enumerate(selected):
            print(f"  [{i}] Group {h['group_id']:5d} | "
                  f"M200={h['M200']:.3e} | R200={h['R200']:.1f} ckpc/h")

        return selected

    def download_halo_particles(self, halo_meta, output_dir="data",
                                 particle_types=("dm", "stars")):
        """
        Download particle data for one halo.

        Uses the TNG "cutout" API which returns only particles within
        a sphere around the halo centre — much smaller than a full snapshot.

        Parameters
        ----------
        halo_meta     : dict from get_top_halos()
        output_dir    : where to save the HDF5 file
        particle_types: which particle types to download

        Returns
        -------
        str : path to saved HDF5 file
        """
        os.makedirs(output_dir, exist_ok=True)
        gid      = halo_meta["group_id"]
        out_path = os.path.join(output_dir, f"halo_{gid:05d}.hdf5")

        if os.path.exists(out_path):
            print(f"  Halo {gid} already downloaded: {out_path}")
            return out_path

        print(f"\nDownloading halo {gid} particles...")

        # Build cutout request
        # TNG cutout API: download all particles within Group_R_Crit200
        cutout_url = (f"{self.base_url}/halos/{gid}/cutout.hdf5?"
                      f"dm=Coordinates,Masses,Velocities&"
                      f"stars=Coordinates,Masses,Velocities&"
                      f"gas=Coordinates,Masses,Velocities,Density")

        r = self._get(cutout_url, stream=True)

        # Stream to disk
        size = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                size += len(chunk)
                print(f"\r  Downloaded: {size/1e6:.1f} MB", end="", flush=True)

        print(f"\n  Saved: {out_path} ({size/1e6:.1f} MB)")
        return out_path


class TNGHaloReader:
    """
    Reads a downloaded TNG HDF5 cutout file and returns clean numpy arrays.

    Usage:
        reader = TNGHaloReader("data/halo_00000.hdf5")
        dm     = reader.get_dark_matter()
        stars  = reader.get_stars()
        info   = reader.get_halo_info()
    """

    # TNG HDF5 field names (these are fixed by the simulation format)
    DM_POS_FIELD    = "PartType1/Coordinates"
    DM_VEL_FIELD    = "PartType1/Velocities"
    DM_MASS_FIELD   = "PartType1/Masses"
    STAR_POS_FIELD  = "PartType4/Coordinates"
    STAR_VEL_FIELD  = "PartType4/Velocities"
    STAR_MASS_FIELD = "PartType4/Masses"
    GAS_POS_FIELD   = "PartType0/Coordinates"
    GAS_MASS_FIELD  = "PartType0/Masses"
    GAS_DENS_FIELD  = "PartType0/Density"

    def __init__(self, hdf5_path):
        if not H5PY_AVAILABLE:
            raise ImportError("pip install h5py")
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        self.path = hdf5_path

        # Read header to get cosmological params
        with h5py.File(hdf5_path, "r") as f:
            self.header = dict(f["Header"].attrs)
            self.scale_factor = self.header.get("Time", 1.0)
            self.h_cosmo      = self.header.get("HubbleParam", 0.6774)
            self.redshift     = self.header.get("Redshift", 0.0)
            self.box_size     = self.header.get("BoxSize", 0.0)   # ckpc/h

        print(f"HDF5 reader opened: {hdf5_path}")
        print(f"  z={self.redshift:.3f}  a={self.scale_factor:.4f}  "
              f"h={self.h_cosmo:.4f}")

    def _read_field(self, field_path, f):
        """Safely read an HDF5 field, return None if missing."""
        if field_path in f:
            return f[field_path][:]
        return None

    def get_dark_matter(self):
        """
        Read dark matter particle data.
        Returns dict with raw TNG arrays (NOT yet unit-converted).
        """
        with h5py.File(self.path, "r") as f:
            pos  = self._read_field(self.DM_POS_FIELD,  f)
            vel  = self._read_field(self.DM_VEL_FIELD,  f)
            mass = self._read_field(self.DM_MASS_FIELD, f)

        # DM particles in TNG often have equal mass (stored as scalar in header)
        if mass is None and pos is not None:
            dm_mass_per_part = self.header.get("MassTable", [0]*6)[1]
            mass = np.full(pos.shape[0], dm_mass_per_part)

        n = pos.shape[0] if pos is not None else 0
        print(f"  DM particles: {n:,}")

        return {"pos": pos, "vel": vel, "mass": mass, "n": n}

    def get_stars(self):
        """Read stellar particle data."""
        with h5py.File(self.path, "r") as f:
            pos  = self._read_field(self.STAR_POS_FIELD,  f)
            vel  = self._read_field(self.STAR_VEL_FIELD,  f)
            mass = self._read_field(self.STAR_MASS_FIELD, f)

        n = pos.shape[0] if pos is not None else 0
        print(f"  Star particles: {n:,}")
        return {"pos": pos, "vel": vel, "mass": mass, "n": n}

    def get_gas(self):
        """Read gas cell data."""
        with h5py.File(self.path, "r") as f:
            pos  = self._read_field(self.GAS_POS_FIELD,  f)
            mass = self._read_field(self.GAS_MASS_FIELD, f)
            dens = self._read_field(self.GAS_DENS_FIELD, f)

        n = pos.shape[0] if pos is not None else 0
        print(f"  Gas cells: {n:,}")
        return {"pos": pos, "mass": mass, "density": dens, "n": n}

    def get_halo_info(self):
        """Return header metadata."""
        return {
            "path"        : self.path,
            "redshift"    : self.redshift,
            "scale_factor": self.scale_factor,
            "h"           : self.h_cosmo,
            "box_size_ckpch": self.box_size,
        }


class HaloProcessor:
    """
    Takes raw TNG particle arrays and produces clean, centred,
    unit-converted arrays ready for the neural network.

    Steps performed:
        1. Convert units (ckpc/h → kpc, 1e10 M_sun/h → M_sun)
        2. Centre on halo potential minimum (most bound particle)
        3. Remove unbound / outlier particles beyond 3 * R200
        4. Return cleaned arrays with diagnostics
    """

    def __init__(self, tng_units):
        """
        tng_units : TNGUnits instance for this snapshot
        """
        self.units = tng_units

    def process(self, raw_dm, raw_stars=None,
                 halo_centre_raw=None, r200_raw=None):
        """
        Full processing pipeline for one halo.

        Parameters
        ----------
        raw_dm          : dict from TNGHaloReader.get_dark_matter()
        raw_stars       : dict from TNGHaloReader.get_stars() (optional)
        halo_centre_raw : (3,) array in ckpc/h — halo centre of mass
        r200_raw        : float in ckpc/h — virial radius

        Returns
        -------
        dict with clean physical-unit arrays and metadata
        """
        u = self.units

        # ── 1. Convert DM positions and masses ───────────────────────────────
        dm_pos_kpc  = u.convert_positions(raw_dm["pos"])       # (N, 3) kpc
        dm_mass_sun = u.convert_masses(raw_dm["mass"])         # (N,)   M_sun
        dm_vel_kms  = u.convert_velocities(raw_dm["vel"])      # (N, 3) km/s

        # ── 2. Centre on halo ─────────────────────────────────────────────────
        if halo_centre_raw is not None:
            centre_kpc  = u.convert_positions(halo_centre_raw)
            dm_pos_kpc  = u.centre_positions(dm_pos_kpc, centre_kpc)
        else:
            # Use centre of mass as fallback
            total_mass   = dm_mass_sun.sum()
            centre_kpc   = (dm_pos_kpc * dm_mass_sun[:, None]).sum(0) / total_mass
            dm_pos_kpc   = u.centre_positions(dm_pos_kpc, centre_kpc)

        # ── 3. Cut to virial radius ───────────────────────────────────────────
        if r200_raw is not None:
            r200_kpc = u.virial_radius_kpc(r200_raw)
            radii    = np.linalg.norm(dm_pos_kpc, axis=1)
            mask     = radii <= 2.0 * r200_kpc    # keep particles within 2*R200
            dm_pos_kpc  = dm_pos_kpc[mask]
            dm_mass_sun = dm_mass_sun[mask]
            dm_vel_kms  = dm_vel_kms[mask]
            r200_kpc_out = r200_kpc
            print(f"  DM particles within 2*R200: {mask.sum():,} / {len(mask):,}")
        else:
            r200_kpc_out = np.linalg.norm(dm_pos_kpc, axis=1).max() / 2.0

        # ── 4. Process stars (if provided) ────────────────────────────────────
        stars_out = None
        if raw_stars is not None and raw_stars["pos"] is not None:
            st_pos = u.convert_positions(raw_stars["pos"])
            st_pos = u.centre_positions(st_pos, centre_kpc if halo_centre_raw is None
                                         else u.convert_positions(halo_centre_raw))
            st_mass = u.convert_masses(raw_stars["mass"])
            st_vel  = u.convert_velocities(raw_stars["vel"])
            if r200_raw is not None:
                st_r  = np.linalg.norm(st_pos, axis=1)
                st_mask = st_r <= 2.0 * r200_kpc_out
                st_pos  = st_pos[st_mask]
                st_mass = st_mass[st_mask]
                st_vel  = st_vel[st_mask]
            stars_out = {"pos_kpc": st_pos, "mass_msun": st_mass,
                         "vel_kms": st_vel}

        total_mass = dm_mass_sun.sum()
        print(f"  Total DM mass : {total_mass:.3e} M_sun")
        print(f"  R200          : {r200_kpc_out:.1f} kpc")
        print(f"  N_DM (clean)  : {dm_pos_kpc.shape[0]:,}")

        return {
            "dm_pos_kpc"   : dm_pos_kpc,
            "dm_mass_msun" : dm_mass_sun,
            "dm_vel_kms"   : dm_vel_kms,
            "stars"        : stars_out,
            "r200_kpc"     : r200_kpc_out,
            "total_dm_mass": total_mass,
            "n_dm"         : dm_pos_kpc.shape[0],
        }
