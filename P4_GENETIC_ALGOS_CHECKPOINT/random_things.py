def stupid_func(param_1, name):
    print(f"The param 1 provided was {param_1}")
    print(f"The name provided was {name}")

param_dict = {"name": "Zac Efron",
              "param_1": 12345}

stupid_func(**param_dict)

