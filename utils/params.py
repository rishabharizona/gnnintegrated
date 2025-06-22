def get_params():
    paramname = {'diversify': [' --latent_domain_num ',
                               ' --alpha1 ',  ' --alpha ', ' --lam ', ' --use_gnn ', ' --gnn_hidden_dim ', ' --gnn_output_dim ']}
    paramlist = {
        'diversify': [[2, 3, 5, 10, 20], [0.1, 0.5, 1], [0.1, 1, 10], [0], [[1, 150], [3, 50], [5, 30], [10, 15], [30, 5]], [0, 1], [16, 32, 64], [64, 128, 256]]
    }
    return paramname, paramlist
