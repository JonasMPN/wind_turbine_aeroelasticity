from BEM import BEM

bem = BEM(data_root="../data/results", file_airfoil="../data/input/polar.xlsx")
bem.set_constants(rotor_radius=50,
                  root_radius=0.2*50,
                  n_blades=3,
                  air_density=1.225)
results = bem.solve(wind_speed=10,
                    tip_speed_ratio=10,
                    pitch=0,
                    start_radius=0.25*50,
                    resolution=10,
                    max_convergence_error=1e-10,
                    verbose=True)
