from CRO_SL import *

def test_cro():
    substrates_int = [
        SubstrateInt("DE/best/1"),
        SubstrateInt("DE/rand/1"),
        SubstrateInt("DE/rand/2"),
        SubstrateInt("DE/best/2"), 
        SubstrateInt("DE/current-to-best/1"),
        SubstrateInt("DE/current-to-rand/1"),
        SubstrateInt("AddOne"),
        SubstrateInt("DGauss"),
        SubstrateInt("Perm"),
        SubstrateInt("1point"),
        SubstrateInt("2point"),
        SubstrateInt("Multipoint"),
        SubstrateInt("Xor")
    ]

    substrates_real = [
        #SubstrateReal("SBX", {"F":0.5}),
        #SubstrateReal("Perm", {"F":0.3}),
        #SubstrateReal("1point"),
        #SubstrateReal("2point"),
        #SubstrateReal("Multipoint"),
        #SubstrateReal("BLXalpha", {"F":0.8}),
        #SubstrateReal("Rand"),
        SubstrateReal("DE/best/1", {"F":0.5, "Pr":0.8}),
        SubstrateReal("DE/rand/1", {"F":0.7, "Pr":0.8}),
        SubstrateReal("DE/best/2", {"F":0.7, "Pr":0.8}),
        SubstrateReal("DE/rand/2", {"F":0.5, "Pr":0.8}),
        SubstrateReal("DE/current-to-best/1", {"F":0.7, "Pr":0.8}),
        SubstrateReal("DE/current-to-rand/1", {"F":0.7, "Pr":0.8}),
        #SubstrateReal("HS", {"F":0.5, "Pr":0.8}),
        #SubstrateReal("SA", {"F":0.14, "temp_ch":10, "iter":20}),
        #SubstrateReal("Gauss", {"F":0.04})
    ]
    
    params = {
        "ReefSize": 100,
        "rho": 0.6,
        "Fb": 0.98,
        "Fd": 1,
        "Pd": 0.1,
        "k": 3,
        "K": 20,
        "group_subs": True,

        "stop_cond": "neval",
        "time_limit": 4000.0,
        "Ngen": 3500,
        "Neval": 6e5,
        "fit_target": 1000,

        "verbose": True,
        "v_timer": 5,

        "dynamic": True,
        "method": "fitness",
        "dyn_metric": "best",
        "prob_amp": 0.07
    }

    #objfunc = MaxOnes(1000)
    #objfunc = MaxOnesReal(1000)
    #objfunc = Sphere(1000)
    #objfunc = Test1(30)
    objfunc = Rosenbrock(30)
    #objfunc = Rastrigin(30)

    #c = CRO_SL(objfunc, substrates_int, params)
    c = CRO_SL(objfunc, substrates_real, params)
    ind, fit = c.optimize()
    print(ind)
    c.display_report()

def thity_runs():
    substrates_real = [
        SubstrateReal("DE/rand/1", {"F":0.7, "Pr":0.8}),
        SubstrateReal("DE/best/2", {"F":0.7, "Pr":0.8}),
        SubstrateReal("DE/current-to-best/1", {"F":0.7, "Pr":0.8}),
        SubstrateReal("DE/current-to-rand/1", {"F":0.7, "Pr":0.8})
    ]

    params = {
        "ReefSize": 100,
        "rho": 0.6,
        "Fb": 0.98,
        "Fd": 1,
        "Pd": 0.1,
        "k": 3,
        "K": 20,
        "group_subs": True,

        "stop_cond": "neval",
        "time_limit": 4000.0,
        "Ngen": 3500,
        "Neval": 3e5,
        "fit_target": 1000,

        "verbose": False,
        "v_timer": 10,

        "dynamic": False,
        "method": "fitness",
        "dyn_metric": "best",
        "prob_amp": 0.05
    }

    n_coord = 30
    n_runs = 10
    
    combination_DE = [
        [0,1,2,3],
        [0,1,2],
        [0,1,3],
        [0,2,3],
        [1,2,3],
        [0,1],
        [0,2],
        [0,3],
        [1,2],
        [1,3],
        [2,3],
        [0],
        [1],
        [2],
        [3]
    ]

    for comb in combination_DE:
        substrates_filtered = [substrates_real[i] for i in range(4) if i in comb]
        fit_list = []
        for i in range(n_runs):
            c = CRO_SL(Rosenbrock(n_coord), substrates_filtered, params)
            _, fit = c.optimize()
            fit_list.append(fit)
            #print(f"Run {i+1} {comb} ended")
            #c.display_report(show_plots=False)
        fit_list = np.array(fit_list)
        print(f"\nRESULTS {comb}:")
        print(f"min: {fit_list.min():e}; mean: {fit_list.mean():e}; std: {fit_list.std():e}")

def main():
    #thity_runs()
    test_cro()

if __name__ == "__main__":
    main()