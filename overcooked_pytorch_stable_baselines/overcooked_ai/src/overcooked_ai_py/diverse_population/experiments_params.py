def set_layout_params(args):
    """
    Layout specific parameters are set
    """
    if args.layout_name == "coordination_ring":
        args.eval_stop_threshold = 185

    if args.layout_name == "cramped_room":
        args.eval_stop_threshold = 20  # 195

    if args.layout_name == "asymmetric_advantages":
        args.eval_stop_threshold = 205

    if args.layout_name == "forced_coordination":
        args.eval_stop_threshold = 160 #150

    if args.layout_name == "counter_circuit_o_1order":
        args.eval_stop_threshold = 123
        args.divergent_check_timestep = 2.6e6

    if args.layout_name == "diagonal":
        args.eval_stop_threshold = 2

    if args.layout_name == "simple_o":
        args.eval_stop_threshold = 10

    if args.layout_name == "cramped2":
        args.eval_stop_threshold = 2
