"""XBPM bump acquisition helpers extracted from the scan_bumps notebook.

This module contains device wrappers, orbit bump helpers, and orchestration
functions for running bump scans at different beamlines. It removes the
implicit globals from the notebook and exposes callable functions for use
by other parts of XBPM-bumps or external scripts.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Iterable, NamedTuple, Optional, Sequence, Tuple

import epics
import numpy as np
from mathphys.functions import save
from siriuspy.clientconfigdb import ConfigDBClient
from siriuspy.devices import APU, SOFB, VPU, CurrInfoSI


def configure_epics_addr(addrs: Sequence[str]) -> None:
    """Append EPICS CA addresses to the current environment list."""
    if not addrs:
        return
    current = os.environ.get("EPICS_CA_ADDR_LIST", "")
    addition = " ".join(addrs)
    os.environ["EPICS_CA_ADDR_LIST"] = f"{current} {addition}".strip()


class XbpmDevice:
    """Generic XBPM device helper."""

    def __init__(
        self,
        prefix: str,
        amp_pvs: Sequence[str],
        range_pvs: Sequence[Optional[str]],
        unit_pvs: Optional[Sequence[Optional[str]]] = None,
    ):
        """Initialize XBPM device with PVs."""
        self.prefix     = prefix
        self._amp_pvs   = amp_pvs
        self._range_pvs = range_pvs
        self._unit_pvs  = unit_pvs or [None] * len(amp_pvs)

    def _amp(self, idx: int):
        unit_pv = self._unit_pvs[idx]
        return epics.caget(self.prefix + self._amp_pvs[idx]), (
            epics.caget(unit_pv) if unit_pv else 0
        )

    def _rng(self, idx: int):
        pv = self._range_pvs[idx]
        return epics.caget(pv) if pv else 0

    def amp_a(self):
        """Get amplitude A."""
        return self._amp(0)

    def amp_b(self):
        """Get amplitude B."""
        return self._amp(1)

    def amp_c(self):
        """Get amplitude C."""
        return self._amp(2)

    def amp_d(self):
        """Get amplitude D."""
        return self._amp(3)

    def range_a(self):
        """Get range A."""
        return self._rng(0)

    def range_b(self):
        """Get range B."""
        return self._rng(1)

    def range_c(self):
        """Get range C."""
        return self._rng(2)

    def range_d(self):
        """Get range D."""
        return self._rng(3)

    ampA = amp_a  # noqa: N815
    ampB = amp_b  # noqa: N815
    ampC = amp_c  # noqa: N815
    ampD = amp_d  # noqa: N815
    rangeA = range_a  # noqa: N815
    rangeB = range_b  # noqa: N815
    rangeC = range_c  # noqa: N815
    rangeD = range_d  # noqa: N815


class XbpmMnc(XbpmDevice):
    """XBPM for MANACÁ beamline."""

    def __init__(self, xbpm_nr: int = 1):
        """Initialize XBPM for MANACÁ beamline."""
        super().__init__(
            prefix=f"SI-09SAFE:DI-PBPM-{xbpm_nr}",
            amp_pvs=[":AmplA-Mon", ":AmplB-Mon", ":AmplC-Mon", ":AmplD-Mon"],
            range_pvs=[None, None, None, None],
        )


class XbpmMgn(XbpmDevice):
    """XBPM for MOGNO beamline."""

    def __init__(self, xbpm_nr: int = 1):
        """Initialize XBPM for MOGNO beamline."""
        super().__init__(
            prefix=f"SI-10BCFE:DI-PBPM-{xbpm_nr}",
            amp_pvs=[":AmplA-Mon", ":AmplB-Mon", ":AmplC-Mon", ":AmplD-Mon"],
            range_pvs=[None, None, None, None],
        )


class XbpmCat(XbpmDevice):
    """XBPM for CATERETE."""

    def __init__(self, xbpm_nr: int = 1):
        """Initialize XBPM for CATERETE."""
        _ = xbpm_nr  # kept for API parity
        super().__init__(
            prefix="CAT:FE:PICO01",
            amp_pvs=[
                ":Current1:EngValue",
                ":Current2:EngValue",
                ":Current3:EngValue",
                ":Current4:EngValue",
            ],
            range_pvs=[
                "CAT:FE:PICO01:Current1:Range_RBV",
                "CAT:FE:PICO01:Current2:Range_RBV",
                "CAT:FE:PICO01:Current3:Range_RBV",
                "CAT:FE:PICO01:Current4:Range_RBV",
            ],
            unit_pvs=[
                "CAT:FE:PICO01:Current1:EngUnit",
                "CAT:FE:PICO01:Current2:EngUnit",
                "CAT:FE:PICO01:Current3:EngUnit",
                "CAT:FE:PICO01:Current4:EngUnit",
            ],
        )


class XbpmCnb(XbpmDevice):
    """XBPM for CARNAUBA."""

    def __init__(self, xbpm_nr: int = 1):
        """Initialize XBPM for CARNAUBA."""
        _ = xbpm_nr
        super().__init__(
            prefix="CNB:FE:PICO01",
            amp_pvs=[
                ":Current1:EngValue",
                ":Current2:EngValue",
                ":Current3:EngValue",
                ":Current4:EngValue",
            ],
            range_pvs=[
                "CNB:FE:PICO01:Current1:Range_RBV",
                "CNB:FE:PICO01:Current2:Range_RBV",
                "CNB:FE:PICO01:Current3:Range_RBV",
                "CNB:FE:PICO01:Current4:Range_RBV",
            ],
            unit_pvs=[
                "CNB:FE:PICO01:Current1:EngUnit",
                "CNB:FE:PICO01:Current2:EngUnit",
                "CNB:FE:PICO01:Current3:EngUnit",
                "CNB:FE:PICO01:Current4:EngUnit",
            ],
        )


# Backwards compatibility with notebook naming (was snake_case with caps).
xbpmMNC = XbpmMnc  # noqa: N816
xbpmMGN = XbpmMgn  # noqa: N816
xbpmCAT = XbpmCat  # noqa: N816
xbpmCNB = XbpmCnb  # noqa: N816


def move_vpu_gap(vpu: VPU, gap: float, timeout: float,
                 verbose: bool = False) -> bool:
    """Move VPU to the specified gap position.

    Args:
        vpu (VPU): VPU device instance.
        gap (float): Target gap position in mm.
        timeout (float): Timeout for the move command in seconds.
        verbose (bool): If True, print detailed status messages.

    Returns:
        bool: True if the move was successful, False otherwise.
    """
    vpu.set_gap(gap)
    time.sleep(0.5)
    if vpu.cmd_move_start(timeout):
        time.sleep(0.5)
        if verbose:
            print("Undulator is moving...")
        while vpu.is_moving:
            time.sleep(0.1)
            if verbose:
                print(f"Current gap {vpu.gap_mon:.3f} mm.", end="\r")
        if verbose:
            print(f"Gap {vpu.gap:.3f} mm reached.")
        return True
    print("Error while cmd_move_start.")
    return False


def move_apu_phase(apu: APU, phase: float, timeout: float,
                   verbose: bool = False) -> bool:
    """Move APU to the specified phase position.

    Args:
        apu (APU): APU device instance.
        phase (float): Target phase position in mm.
        timeout (float): Timeout for the move command in seconds.
        verbose (bool): If True, print detailed status messages.

    Returns:
        bool: True if the move was successful, False otherwise.
    """
    apu.set_phase(phase)
    time.sleep(0.5)
    if apu.cmd_move_start(timeout):
        time.sleep(0.5)
        if verbose:
            print("Undulator is moving...")
        while apu.is_moving:
            time.sleep(0.1)
            if verbose:
                print(f"Current phase {apu.phase_mon:.3f} mm.", end="\r")
        if verbose:
            print(f"Phase {apu.phase:.3f} mm reached.")
        return True
    print("Error while cmd_move_start.")
    return False


def meas(
    idlist: Sequence,
    xbpmlist: Sequence,
    names: Sequence[str],
    sofb: SOFB,
    currinfo: CurrInfoSI,
    subsec: str = "09SA",
    agx: float = 0,
    agy: float = 0,
    psx: float = 0,
    psy: float = 0,
    nr_meas: int = 10,
    tinterval: float = 1.0,
    save_pickle: bool = True,
) -> Tuple[dict, dict, dict]:
    """Measure XBPM signals and machine parameters.

    Args:
        idlist: List of device instances [VPU, APU].
        xbpmlist: List of XBPM device instances.
        names: List of names corresponding to each XBPM.
        sofb (SOFB): SOFB instance to get orbit data.
        currinfo (CurrInfoSI): CurrInfoSI instance to get current data.
        subsec (str, optional): Subsection where the bump is implemented.
        agx (float, optional): Horizontal angle for the bump in urad.
        agy (float, optional): Vertical angle for the bump in urad.
        psx (float, optional): Horizontal position offset for the bump in um.
        psy (float, optional): Vertical position offset for the bump in um.
        nr_meas (int, optional): Number of measurements to perform.
        tinterval (float, optional): Time interval between measurements (s).
        save_pickle (bool, optional): Persist acquisition to pickle like the
            notebook.

    Returns:
        tuple: (blades, idinfo, machinfo).
    """
    blades = {}
    for name, xbpm in zip(names, xbpmlist, strict=False):
        blades[name] = {
            "prefix": xbpm.prefix,
            "A_val": [],
            "B_val": [],
            "C_val": [],
            "D_val": [],
            "A_range": [],
            "B_range": [],
            "C_range": [],
            "D_range": [],
        }

    for _ in np.arange(nr_meas):
        for name, xbpm in zip(names, xbpmlist, strict=False):
            blades[name]["A_val"].append(xbpm.amp_a())
            blades[name]["B_val"].append(xbpm.amp_b())
            blades[name]["C_val"].append(xbpm.amp_c())
            blades[name]["D_val"].append(xbpm.amp_d())
            blades[name]["A_range"].append(xbpm.range_a())
            blades[name]["B_range"].append(xbpm.range_b())
            blades[name]["C_range"].append(xbpm.range_c())
            blades[name]["D_range"].append(xbpm.range_d())
        time.sleep(tinterval)

    vpu_cnb = idlist[0]
    apu_mnc = idlist[1]

    idinfo = {
        "cnb": getattr(vpu_cnb, "gap_mon", None),
        "mnc": getattr(apu_mnc, "phase_mon", None),
    }

    machinfo = {
        "current": currinfo.current,
        "agx": agx,
        "agy": agy,
        "posx": psx,
        "posy": psy,
        "orbx": sofb.orbx,
        "orby": sofb.orby,
    }

    data = (blades, idinfo, machinfo)

    if save_pickle:
        fname = "xbpm_acq_subsec:"
        fname += f"{subsec}_agx:{agx:03.0f}_agy:{agy:03.0f}_"
        fname += f"posx:{psx:03.0f}_posy:{psy:03.0f}_"
        fname += "time_" + str(time.time()) + ".pickle"
        save(data, fname)

    return data


def implement_bump(
    sofb: SOFB,
    subsec: str = "09SA",
    refx: Optional[np.ndarray] = None,
    refy: Optional[np.ndarray] = None,
    agx: float = 0,
    agy: float = 0,
    psx: float = 0,
    psy: float = 0,
    tol_orb: float = 3,
) -> None:
    """Implement a local bump in the orbit using SOFB.

    Parameters
    ----------
    sofb : SOFB
        The SOFB instance to use for orbit correction.
    subsec : str, optional
        The subsection where the bump will be implemented.
    refx : np.ndarray, optional
        Reference horizontal orbit (fetches from config DB if None).
    refy : np.ndarray, optional
        Reference vertical orbit (fetches from config DB if None).
    agx : float, optional
        Horizontal angle for the bump in urad.
    agy : float, optional
        Vertical angle for the bump in urad.
    psx : float, optional
        Horizontal position offset for the bump in um.
    psy : float, optional
        Vertical position offset for the bump in um.
    tol_orb : float, optional
        Tolerance for orbit correction in um.
    """

    def get_rms(refx_arr, refy_arr, idx):
        dorbx = sofb.orbx - refx_arr
        dorby = sofb.orby - refy_arr
        dorbx = dorbx[idx]
        dorby = dorby[idx]
        return np.hstack([dorbx, dorby]).std()

    max_tol_orb = 10

    if refx is None or refy is None:
        clt = ConfigDBClient(config_type="si_orbit")
        ref_orb = clt.get_config_value("ref_orb")
        refx = np.array(ref_orb["x"])
        refy = np.array(ref_orb["y"])

    orbx, orby = sofb.si_calculate_bumps(
        refx, refy, subsec=subsec, agx=agx, agy=agy, psx=psx, psy=psy
    )

    dummy, _ = sofb.si_calculate_bumps(refx, refy, subsec=subsec, agx=10)
    idx = ~np.isclose(dummy, refx)
    strt = idx.nonzero()[0][0]

    sofb.refx = orbx
    sofb.refy = orby
    enbl = np.ones(orbx.size, dtype=bool)
    nr_bpms = 4
    enbl[strt - nr_bpms : strt] = False
    enbl[strt + 2 : strt + 2 + nr_bpms] = False
    sofb.bpmxenbl = enbl
    sofb.bpmyenbl = enbl

    rms_residue = tol_orb + 1
    nr_iters = 10
    while rms_residue > tol_orb:
        _ = sofb.correct_orbit_manually(nr_iters=nr_iters, residue=1)
        rms_residue = get_rms(orbx, orby, idx)
        tol_orb *= 1.2
        if tol_orb > max_tol_orb:
            raise ValueError("Could not correct orbit.")


def restore_sofb_reforb(sofb: SOFB, bpmxenbl, bpmyenbl) -> None:
    """Restore SOFB reference orbit and BPM enable flags."""
    clt = ConfigDBClient(config_type="si_orbit")
    ref_orb = clt.get_config_value("ref_orb")
    refx = np.array(ref_orb["x"])
    refy = np.array(ref_orb["y"])
    sofb.refx = refx
    sofb.refy = refy
    sofb.bpmxenbl = bpmxenbl
    sofb.bpmyenbl = bpmyenbl


def is_beam_alive(currinfo: CurrInfoSI, sofb: SOFB, idlist: Sequence) -> bool:
    """Return True if beam is stored, otherwise restore devices."""
    if currinfo.storedbeam:
        return True
    print("Beam is dead!")
    restore_sofb_reforb(sofb, sofb.bpmxenbl, sofb.bpmyenbl)
    move_apu_phase(idlist[1], 11, timeout=10)
    move_vpu_gap(idlist[0], 80, timeout=10)
    return False


def stop_meas_func(stop_event: Optional[threading.Event],
                   pause_event: Optional[threading.Event],
                   sofb: SOFB,
                   idlist: Sequence) -> bool:
    """Check stop flag, perform safe restore, return True if stopped."""
    if stop_event and stop_event.is_set():
        restore_sofb_reforb(sofb, sofb.bpmxenbl, sofb.bpmyenbl)
        sofb.correct_orbit_manually(10, 2)
        move_apu_phase(idlist[1], 11, timeout=10)
        move_vpu_gap(idlist[0], 80, timeout=10)
        if pause_event:
            pause_event.clear()
        return True
    return False


def _angle_grid(angsx: Iterable[float], angsy: Iterable[float]):
    """Yield (agx, agy) pairs with alternating sign on vertical list."""
    for idx, agx in enumerate(angsx):
        row = angsy if idx % 2 == 0 else (-np.asarray(angsy))
        for agy in row:
            yield agx, float(agy)


def _wait_if_paused(stop_event: Optional[threading.Event],
                    pause_event: Optional[threading.Event]) -> None:
    """Block while pause_event is set; exit early if stop_event is set."""
    if not pause_event:
        return
    while pause_event.is_set():
        time.sleep(2)
        if stop_event and stop_event.is_set():
            break


def _should_continue(currinfo: CurrInfoSI,
                     sofb: SOFB,
                     idlist: Sequence,
                     stop_event: Optional[threading.Event],
                     pause_event: Optional[threading.Event]) -> bool:
    """Return False when beam is dead or stop flag is set."""
    if not is_beam_alive(currinfo, sofb, idlist):
        return False
    if stop_meas_func(stop_event, pause_event, sofb, idlist):
        return False
    return True


def do_bumps(
    angsx: Iterable[float],
    angsy: Iterable[float],
    subsec: str,
    idlist: Sequence,
    currinfo: CurrInfoSI,
    sofb: SOFB,
    xbpmlist: Sequence,
    names: Sequence[str],
    nr_meas: int = 10,
    tinterval: float = 1.0,
    stop_event: Optional[threading.Event] = None,
    pause_event: Optional[threading.Event] = None,
) -> None:
    """Iterate over angle grid, apply bumps, and acquire measurements.

    Args:
        angsx: Iterable of horizontal angles in urad.
        angsy: Iterable of vertical angles in urad.
        subsec: Subsection where the bump is implemented.
        idlist: List of device instances [VPU, APU].
        currinfo: CurrInfoSI instance to get current data.
        sofb: SOFB instance to use for orbit correction.
        xbpmlist: List of XBPM device instances.
        names: List of names corresponding to each XBPM.
        nr_meas: Number of measurements to perform at each bump.
        tinterval: Time interval between measurements (s).
        stop_event: Optional threading.Event to signal stopping the process.
        pause_event: Optional threading.Event to signal pausing the process.

    Returns:
        None
    """
    for agx, agy in _angle_grid(angsx, angsy):
        if not _should_continue(currinfo, sofb, idlist,
                                stop_event, pause_event):
            break

        print(f"agx:{agx:.0f}    agy:{agy:.0f}")
        try:
            implement_bump(sofb=sofb, subsec=subsec, agx=agx, agy=agy)
            meas(
                idlist,
                xbpmlist,
                names,
                sofb,
                currinfo,
                subsec=subsec,
                agx=agx,
                agy=agy,
                nr_meas=nr_meas,
                tinterval=tinterval,
            )
            time.sleep(0.1)
            _wait_if_paused(stop_event, pause_event)
        except ValueError as err:
            print(err)
            continue
    restore_sofb_reforb(sofb, sofb.bpmxenbl, sofb.bpmyenbl)


def _build_default_sofb() -> Tuple[SOFB, CurrInfoSI, np.ndarray, np.ndarray]:
    """Instantiate SOFB and CurrInfoSI with default reference orbit."""
    sofb = SOFB(SOFB.DEVICES.SI)
    currinfo = CurrInfoSI()
    clt = ConfigDBClient(config_type="si_orbit")
    ref_orb = clt.get_config_value("ref_orb")
    refx = np.array(ref_orb["x"])
    refy = np.array(ref_orb["y"])
    return sofb, currinfo, refx, refy


class DeviceContext(NamedTuple):
    """Bundle of shared devices to reuse across runs."""

    sofb: SOFB
    currinfo: CurrInfoSI
    apu: APU
    vpu: VPU
    bpmxenbl: np.ndarray
    bpmyenbl: np.ndarray


def prepare_device_context() -> DeviceContext:
    """Initialize devices once and turn off SOFB autocorr.

    Devices: SOFB, CurrInfoSI, APU, VPU.
    """
    sofb, currinfo, _refx, _refy = _build_default_sofb()
    apu = APU(APU.DEVICES.APU22_09SA)
    vpu = VPU(VPU.DEVICES.VPU29_06SB)

    sofb.cmd_turn_off_autocorr()
    bpmxenbl = sofb.bpmxenbl
    bpmyenbl = sofb.bpmyenbl
    return DeviceContext(sofb, currinfo, apu, vpu, bpmxenbl, bpmyenbl)


def run_manaca_measurement(
    npoints: int = 11,
    ang_limit: float = 20.0,
    nr_meas: int = 10,
    tinterval: float = 1.0,
    phase: float = 0,
    stop_event: Optional[threading.Event] = None,
    pause_event: Optional[threading.Event] = None,
    ctx: Optional[DeviceContext] = None,
    restore_after: bool = True,
) -> None:
    """Run MANACA bump grid synchronously."""
    if ctx is None:
        ctx = prepare_device_context()
    sofb, currinfo, apu, vpu, bpmxenbl, bpmyenbl = ctx

    move_apu_phase(apu, phase, timeout=30)

    angsx = np.linspace(-ang_limit, ang_limit, npoints)
    angsy = np.linspace(-ang_limit, ang_limit, npoints)

    xbpmlist = [XbpmMnc(1), XbpmMnc(2)]
    names = ["MNC1", "MNC2"]
    idlist = [vpu, apu]

    try:
        do_bumps(
            angsx,
            angsy,
            "09SA",
            idlist,
            currinfo,
            sofb,
            xbpmlist,
            names,
            nr_meas=nr_meas,
            tinterval=tinterval,
            stop_event=stop_event,
            pause_event=pause_event,
        )
    finally:
        if restore_after:
            restore_sofb_reforb(sofb, bpmxenbl, bpmyenbl)


def run_mogno_measurement(
    nr_meas: int = 10,
    tinterval: float = 1.0,
    stop_event: Optional[threading.Event] = None,
    pause_event: Optional[threading.Event] = None,
    ctx: Optional[DeviceContext] = None,
    restore_after: bool = True,
) -> None:
    """Run MOGNO bump scan (vertical only)."""
    if ctx is None:
        ctx = prepare_device_context()
    sofb, currinfo, apu, vpu, bpmxenbl, bpmyenbl = ctx

    angsx = np.array([0])
    angsy = np.linspace(-20, 20, 11)

    xbpmlist = [XbpmMgn(1), XbpmMgn(2)]
    names = ["MGN1", "MGN2"]
    idlist = [vpu, apu]

    try:
        do_bumps(
            angsx,
            angsy,
            "10BC",
            idlist,
            currinfo,
            sofb,
            xbpmlist,
            names,
            nr_meas=nr_meas,
            tinterval=tinterval,
            stop_event=stop_event,
            pause_event=pause_event,
        )
    finally:
        if restore_after:
            restore_sofb_reforb(sofb, bpmxenbl, bpmyenbl)


def run_carnauba_measurement(
    npoints: int = 11,
    ang_limit: float = 20.0,
    nr_meas: int = 10,
    tinterval: float = 1.0,
    gap: float = 0,
    stop_event: Optional[threading.Event] = None,
    pause_event: Optional[threading.Event] = None,
    ctx: Optional[DeviceContext] = None,
    restore_after: bool = True,
) -> None:
    """Run CARNAÚBA bump grid."""
    if ctx is None:
        ctx = prepare_device_context()
    sofb, currinfo, apu, vpu, bpmxenbl, bpmyenbl = ctx

    move_vpu_gap(vpu, gap, timeout=30)

    angsx = np.linspace(-ang_limit, ang_limit, npoints)
    angsy = np.linspace(-ang_limit, ang_limit, npoints)

    xbpmlist = [XbpmCnb(1)]
    names = ["CNB"]
    idlist = [vpu, apu]

    try:
        do_bumps(
            angsx,
            angsy,
            "06SB",
            idlist,
            currinfo,
            sofb,
            xbpmlist,
            names,
            nr_meas=nr_meas,
            tinterval=tinterval,
            stop_event=stop_event,
            pause_event=pause_event,
        )
    finally:
        if restore_after:
            restore_sofb_reforb(sofb, bpmxenbl, bpmyenbl)


def run_caterete_measurement(
    npoints: int = 11,
    ang_limit: float = 20.0,
    nr_meas: int = 10,
    tinterval: float = 1.0,
    gap: float = 0,
    subsec: str = "06SB",
    stop_event: Optional[threading.Event] = None,
    pause_event: Optional[threading.Event] = None,
    ctx: Optional[DeviceContext] = None,
    restore_after: bool = True,
) -> None:
    """Run CATERETÊ bump grid using CAT XBPMs.

    Args mirror the other beamline helpers; subsec defaults to 06SB but can be
    overridden to match the desired lattice subsection.
    """
    if ctx is None:
        ctx = prepare_device_context()
    sofb, currinfo, apu, vpu, bpmxenbl, bpmyenbl = ctx

    move_vpu_gap(vpu, gap, timeout=30)

    angsx = np.linspace(-ang_limit, ang_limit, npoints)
    angsy = np.linspace(-ang_limit, ang_limit, npoints)

    xbpmlist = [XbpmCat(1)]
    names = ["CAT"]
    idlist = [vpu, apu]

    try:
        do_bumps(
            angsx,
            angsy,
            subsec,
            idlist,
            currinfo,
            sofb,
            xbpmlist,
            names,
            nr_meas=nr_meas,
            tinterval=tinterval,
            stop_event=stop_event,
            pause_event=pause_event,
        )
    finally:
        if restore_after:
            restore_sofb_reforb(sofb, bpmxenbl, bpmyenbl)


def start_threaded(target, *args, **kwargs) -> threading.Thread:
    """Run a target function in a daemon thread and return it."""
    thread = threading.Thread(target=target, args=args, kwargs=kwargs)
    thread.daemon = True
    thread.start()
    return thread


__all__ = [
    "configure_epics_addr",
    "XbpmDevice",
    "XbpmMnc",
    "XbpmMgn",
    "XbpmCat",
    "XbpmCnb",
    "DeviceContext",
    "prepare_device_context",
    "move_vpu_gap",
    "move_apu_phase",
    "meas",
    "implement_bump",
    "restore_sofb_reforb",
    "is_beam_alive",
    "stop_meas_func",
    "do_bumps",
    "run_manaca_measurement",
    "run_mogno_measurement",
    "run_carnauba_measurement",
    "run_caterete_measurement",
    "start_threaded",
]
