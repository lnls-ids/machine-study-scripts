#!/usr/bin/env python-sirius
"""Script for running LOCO algorithm."""
import sys
import time

from mathphys.functions import save, load
import numpy as np

import siriuspy.clientconfigdb as servconf
from pymodels import si
import pyaccel as pyac
from apsuite.loco.config import LOCOConfigSI
from apsuite.loco.main import LOCO
from apsuite.loco.report import LOCOReport
from apsuite.optics_analysis.tune_correction import TuneCorr
import sys
import apsuite.commisslib as commisslib


def save_data(fname,
              config,
              initial_model,
              fit_model,
              gbpm, gcorr, rbpm,
              chi_history, energy_shift, residue_history,
              girder_shift, kldelta_history, ksldelta_history):
    """."""
    data = dict(initial_model=initial_model,
                fit_model=fit_model,
                config=config,
                gain_bpm=gbpm,
                gain_corr=gcorr,
                roll_bpm=rbpm,
                energy_shift=energy_shift,
                chi_history=chi_history,
                res_history=residue_history,
                girder_shift=girder_shift,
                kldelta_history=kldelta_history,
                ksldelta_history=ksldelta_history)
    save(data, fname)


def load_data(fname):
    """."""
    sys.modules['apsuite.commissioning_scripts'] = commisslib
    return load(fname)


def move_tunes(model, loco_setup):
    """."""
    tunex_goal = 49 + loco_setup['tunex']
    tuney_goal = 14 + loco_setup['tuney']

    print('--- correcting si tunes...')
    tunecorr = TuneCorr(
        model, 'SI', method='Proportional', grouping='TwoKnobs')
    tunecorr.get_tunes(model)
    print('    tunes init  : ', tunecorr.get_tunes(model))
    tunemat = tunecorr.calc_jacobian_matrix()
    tunecorr.correct_parameters(
           model=model,
           goal_parameters=np.array([tunex_goal, tuney_goal]),
           jacobian_matrix=tunemat, tol=1e-10)
    print('    tunes final : ', tunecorr.get_tunes(model))

def get_idcs(fam_name, mod, idc_ivu):
    idc = np.array(pyac.lattice.find_indices(mod, 'fam_name', fam_name))
    idx = np.argsort(np.abs(idc_ivu-idc))[:2]
    return idc[idx]

def create_loco_config(loco_setup, change_tunes=True):
    """."""
    config = LOCOConfigSI()

    # create nominal model
    # model = si.create_accelerator()
    data = load('./IVU_EMA_ACORM_kparam_24p000/fitting_IVU_EMA_ACORM_kparam_24p000')
    model = data['fit_model']

    # dimension used in the fitting
    config.dim = '6d'

    # change nominal tunes to match the measured values
    if change_tunes:
        move_tunes(model, loco_setup)

    config.model = model

    # initial gains (None set all gains to one and roll to zero)
    config.gain_bpm = None
    config.gain_corr = None
    config.roll_bpm = None

    # # # load previous fitting
    # folder = ''
    # fname = folder + 'fitting_after_b1b2corr'
    # data = load_data(fname)
    # config.model = data['fit_model']
    # config.gain_bpm = data['gain_bpm']
    # config.gain_corr = data['gain_corr']
    # model.cavity_on = True
    # model.radiation_on = True

    if config.dim == '4d':
        config.model.cavity_on = False
        config.model.radiation_on = False
    elif config.dim == '6d':
        config.model.cavity_on = True
        config.model.radiation_on = False

    # Select if LOCO includes dispersion column in matrix, diagonal and
    # off-diagonal elements
    config.use_dispersion = True
    config.use_diagonal = True
    config.use_offdiagonal = True

    # Set if want to fit quadrupoles and dipoles in families instead of
    # individually
    config.use_quad_families = False
    config.use_dip_families = False

    # Add constraints in gradients
    config.constraint_deltakl_step = True * 0
    config.constraint_deltakl_total = False
    config.deltakl_normalization = 1e-3

    config.tolerance_delta = 1e-6
    config.tolerance_overfit = 1e-6

    # Jacobian Inversion method, LevenbergMarquardt requires transpose method

    config.inv_method = LOCOConfigSI.INVERSION.Transpose
    # config.inv_method = LOCOConfigSI.INVERSION.Normal

    # config.min_method = LOCOConfigSI.MINIMIZATION.GaussNewton
    # config.lambda_lm = 0

    config.min_method = LOCOConfigSI.MINIMIZATION.LevenbergMarquardt
    config.lambda_lm = 1e-3
    config.fixed_lambda = False

    # quadrupolar strengths to be included in the fit
    config.fit_quadrupoles = True
    config.fit_sextupoles = False
    config.fit_dipoles = False

    # Select subset of families to be fit, 'None' will include all families by
    # default
    config.quadrupoles_to_fit = None
    config.sextupoles_to_fit = None
    config.skew_quadrupoles_to_fit = config.famname_skewquadset.copy()
    fc2_idx = config.skew_quadrupoles_to_fit.index('FC2')
    config.skew_quadrupoles_to_fit.pop(fc2_idx)
    # config.skew_quadrupoles_to_fit = ['SFA0', 'SDB0', 'SDP0']
    config.dipoles_to_fit = None
    config.update()

    famdata = si.get_family_data(model)

    # TO USE ONLY LOCAL QUADRUPOLES
    # idc_ivu = pyac.lattice.find_indices(model, 'fam_name', 'IVU18')[0]
    # idc_qn = []
    # idc_qn.append(get_idcs('QFB', model, idc_ivu))
    # idc_qn.append(get_idcs('QDB1', model, idc_ivu))
    # idc_qn.append(get_idcs('QDB2', model, idc_ivu))
    # idc_qn = np.array(idc_qn)

    # idc_quads = list()
    # idc_quads.append([idc_qn[0][0]])
    # idc_quads.append([idc_qn[0][1]])
    # idc_quads.append([idc_qn[1][0]])
    # idc_quads.append([idc_qn[1][1]])
    # idc_quads.append([idc_qn[2][0]])
    # idc_quads.append([idc_qn[2][1]])
    # config.quad_indices_kl = idc_quads


    config.quad_indices_kl = famdata['QN']['index']
    config.skew_quad_indices_ksl = famdata['QS']['index']


    # dipolar errors at dipoles
    config.fit_dipoles_kick = False

    # off diagonal elements fitting
    if config.use_offdiagonal:
        # To correct the coupling, set just config.fit_skew_quadrupoles and
        # fit_roll_bpm to True and the others to False
        config.fit_quadrupoles_coupling = False
        config.fit_sextupoles_coupling = False
        config.fit_dipoles_coupling = False

        config.fit_roll_bpm = True
        config.fit_skew_quadrupoles = True
    else:
        config.fit_quadrupoles_coupling = False
        config.fit_sextupoles_coupling = False
        config.fit_dipoles_coupling = False
        config.fit_roll_bpm = False
        config.fit_skew_quadrupoles = False

    config.fit_energy_shift = False

    # BPM and corrector gains (always True by default)
    config.fit_gain_bpm = False
    config.fit_gain_corr = True

    # kicks used in the measurements
    config.delta_kickx_meas = 15e-6  # [rad]
    config.delta_kicky_meas = 1.5 * 15e-6  # [rad]
    config.delta_frequency_meas = 15 * 5  # [Hz]

    # girders shifts
    config.fit_girder_shift = False

    # initial weights

    # BPMs
    config.weight_bpm = None
    # config.weight_bpm = 1/loco_setup['bpm_variation'].flatten()

    # Correctors
    # config.weight_corr = None
    config.weight_corr = np.ones(281)
    # Weight on dispersion column, 280 to be as important as the other columns
    # and 1e6 to match the order of magnitudes. The dispersion factor can be
    # set to force the dispersion fitting harder.
    dispersion_factor = 2
    config.weight_corr[-1] = dispersion_factor * (75/15 * 1e6)

    # Gradients constraints
    # config.weight_deltakl = None

    # Remember Quad_number = 270, Sext_number = 280, Dip_number = 100
    config.weight_deltakl = np.ones(270 + 0*280 + 0*100)

    # # singular values selection method
    # config.svd_method = LOCOConfigSI.SVD.Threshold
    # config.svd_thre = 1e-6
    config.svd_method = LOCOConfigSI.SVD.Selection

    # # When adding the gradient constraint, it is required to remove only the
    # # last singular value
    # # Remember Quad Number = 270, BPM gains = 2 * 160, BPM roll = 1 * 160,
    # # Corrector Gains = 120 + 160, Dip Number = 100, Sext Number = 280,
    # # QS Number = 80
    # config.svd_sel = 270 + 2 * 160 + (120 + 160) + 0 * 280 + 0 * 100 + 80 - 1

    # One can simplify this by setting config.svd_sel = -1, but the way above
    # is far more explict
    config.svd_sel = -1
    config.parallel = True
    return config


def create_loco(
        loco_setup,
        load_jacobian=False, save_jacobian=False,
        change_tunes=True):
    """."""
    config = create_loco_config(loco_setup, change_tunes=change_tunes)
    if 'respmat' in loco_setup:
        orbmat_meas = loco_setup['respmat']
        print('loading respmat from LOCO input file')
    elif 'orbmat_name' in loco_setup:
        client = servconf.ConfigDBClient(config_type='si_orbcorr_respm')
        orbmat_meas = np.array(
            client.get_config_value(name=loco_setup['orbmat_name']))
        print('loading respmat from ServConf')
    else:
        raise Exception('LOCO input file do not have matrix or config name.')

    orbmat_meas[:, -1] *= 1e-6  # convert dispersion column from um to m
    config.goalmat = orbmat_meas

    # swap BPM for test
    # config.goalmat[[26,25], :] = config.goalmat[[25, 26], :]
    # config.goalmat[[160+26, 160+25], :] = config.goalmat[[160+25, 160+26], :]

    # swap CH for test
    # nfig.goalmat[:, [24, 23]] = config.goalmat[:, [23, 24]]

    # swap CV for test
    # config.goalmat[:, [120+32, 120+31]] = config.goalmat[:, [120+31, 120+32]]

    alpha = pyac.optics.get_mcf(config.model)
    rf_frequency = loco_setup['rf_frequency']
    config.measured_dispersion = -1 * alpha * rf_frequency * orbmat_meas[:, -1]

    config.update()
    print('')
    print(config)

    print('[create loco object]')
    loco = LOCO(config=config, save_jacobian_matrices=save_jacobian)

    kl_folder = 'jacobian_KL/' + config.dim
    ksl_folder = 'jacobian_KsL/' + config.dim

    if load_jacobian:
        # Pre-calculated jacobian can be used (remember that these matrices
        # were calculated at nominal tunes nux=49.096 and nuy=14.152)

        # loco.update(
        #     fname_jloco_kl_dip=kl_folder+'/dipoles',
        #     fname_jloco_kl_quad=kl_folder+'/quadrupoles',
        #     fname_jloco_kl_sext=kl_folder+'/sextupoles',
        #     fname_jloco_ksl_dip=ksl_folder+'/dipoles',
        #     fname_jloco_ksl_quad=ksl_folder+'/quadrupoles',
        #     fname_jloco_ksl_sext=ksl_folder+'/sextupoles',
        #     fname_jloco_ksl_skewquad=ksl_folder+'/skew_quadrupoles')

        # # The other option is to read the recent calculated jacobian at
        # # appropriated tunes. LOCO class saves the jacobian files with default
        # # names, please rename it in the folder and use it here

        loco.update(
            fname_jloco_kl_quad='6d_KL_quadrupoles_trims',
            fname_jloco_ksl_skewquad='6d_KsL_skew_quadrupoles',)
    else:
        loco.update()
    return loco


def run_and_save(
        setup_name, file_name, niter,
        load_jacobian=False, save_jacobian=False,
        change_tunes=True):
    """."""
    setup = load_data(setup_name)
    if 'data' in setup.keys():
        setup = setup['data']

    t0 = time.time()
    loco = create_loco(
        setup,
        load_jacobian=load_jacobian, save_jacobian=save_jacobian,
        change_tunes=change_tunes)
    loco.run_fit(niter=niter)
    save_data(
        fname=file_name,
        config=loco.config,
        fit_model=loco.fitmodel,
        gbpm=loco.bpm_gain,
        initial_model=loco.config.model,
        gcorr=loco.corr_gain,
        rbpm=loco.bpm_roll,
        energy_shift=loco.energy_shift,
        chi_history=loco.chi_history,
        residue_history=loco.residue_history,
        girder_shift=loco.girder_shift,
        kldelta_history=loco.kldelta_history,
        ksldelta_history=loco.ksldelta_history)
    dt = time.time() - t0
    print('running time: {:.1f} minutes'.format(dt/60))


fname_setup = sys.argv[1]
if not fname_setup:
    print('must provide a file name.')
suffix = fname_setup[11:]
print(fname_setup)
print(suffix)

fname_fit = 'fitting_' + suffix
run_and_save(
    setup_name=fname_setup,
    file_name=fname_fit,
    niter=20,
    change_tunes=False,
    load_jacobian=False,
    save_jacobian=False)

# report = LOCOReport()
# report.create_report(
#     fname_report='',
#     fname_fit=fname_fit, fname_setup=fname_setup)
