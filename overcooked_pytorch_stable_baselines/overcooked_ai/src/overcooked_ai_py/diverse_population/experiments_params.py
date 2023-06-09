def set_layout_params(args):
    """
    Layout specific parameters are set
    """
    if args.layout_name == "coordination_ring":
        args.eval_stop_threshold = 185

    if args.layout_name == "cramped_room":
        args.eval_stop_threshold = 210  # 195

    if args.layout_name == "asymmetric_advantages":
        args.eval_stop_threshold = 205

    if args.layout_name == "forced_coordination":
        args.eval_stop_threshold = 160 #150

    if args.layout_name == "counter_circuit_o_1order":
        args.eval_stop_threshold = 123
        args.divergent_check_timestep = 2.6e6

    else:
        args.eval_stop_threshold = 150
