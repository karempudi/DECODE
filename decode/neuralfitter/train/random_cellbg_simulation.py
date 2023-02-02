import decode.simulation
import decode.utils

def setup_random_simulation_cells(param):
    """
    Setup a simulation where emitters are overlaid on top of 
    cell background, sampled from the datastore.
    
    Steps involved in the process are as follows.
        1. Define PSF function (load from the calibration file)
        2. Find a way to sample emitter coordinates by 
           looping over the required number of images of corresponding background
        3. Call the appropriate simulation class, that handles the camera model
    """

    psf = decode.utils.calibration_io.SMAPSplineCoefficient(
        calib_file=param.InOut.calibration_file).init_spline(
            xextent=param.Simulation.psf_extent[0],
            yextent=param.Simulation.psf_extent[1],
            img_shape=param.Simulation.img_size,
            device=param.Hardware.device_simulation,
            roi_size=param.Simulation.roi_size,
            roi_auto_center=param.Simulation.roi_auto_center
        )
    
    if param.Simulation.mode in ('acquisition', 'apriori'):
        frame_range_train = (0, param.HyperParameter.pseudo_ds_size)
    
    elif param.Simulation.mode == 'samples':
        frame_range_train = (-((param.HyperParameter.channels_in - 1) // 2),
                             (param.HyperParameter.channels_in - 1) // 2)
    else:
        raise ValueError
    

    # define noise model of the Camera to squish the
    # image through after you do PSF simulation and 
    if param.CameraPreset == 'Perfect':
        noise = decode.simulation.camera.PerfectCamera.parse(param)
    elif param.CameraPreset == 'EMCCD':
        noise = decode.simulation.camera.Photon2Camera.parse(param)
    elif param.CameraPreset == 'SCMOSPhotometrix':
        noise = decode.simulation.camera.SCMOSPhotometrix.parse(param)
    elif param.CameraPreset is not None:
        raise NotImplementedError  

    # use the psf, noise and param to generate the simulation data with a sample() method 
    # on this object
    simulation_train = decode.simulation.simulator.CellSimulation(psf=psf,
                                                                param=param,
                                                                noise=noise,
                                                                frame_range=frame_range_train,
                                                                device=param.Hardware.device_simulation
                                                                )

    frame_range_test = (0, param.TestSet.test_size)

    simulation_test = decode.simulation.simulator.CellSimulation(psf=psf, 
                        param=param,
                        noise=noise,
                        frame_range=frame_range_test,
                        device=param.Hardware.device_simulation)

    return simulation_train, simulation_test