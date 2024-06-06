using MCTS

solver = DPWSolver(n_iterations=20,
                   depth=40,
                   check_repeat_action=true,
                   exploration_constant=0.1,
                   k_action=1.0,
                   alpha_action=0.0,
                   enable_state_pw=false,
                   tree_in_info=true,
                   show_progress=true,
                   estimate_value=value_estimate,
)
