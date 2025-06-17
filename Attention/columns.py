def columns(species, l_max, n_max):
    col = []
    # for i in range(len(species)):
    #     for j in range(len(species)):
    #         for l in range(l_max+1):
    #             for n in range(n_max):
    #                 for n_ in range(n_max):
    #                     if (n_, j) >= (n, i):
    #                         col += [f"p(χ)^({species[i]} {species[j]})_({n} {n_} {l})"]
    # return col


    for i in range(len(species)):
        for j in range(i, len(species)):
            for l in range(l_max+1):
                for n in range(n_max):
                    for n_ in range(n_max):
                        if (n_, j) >= (n, i):
                            col += [f"p(χ)^({species[i]} {species[j]})_({n} {n_} {l})"]
                        else:
                            col += [f"p(χ)^({species[j]} {species[i]})_({n_} {n} {l})"]
                        # count += 1
    return col


import itertools
def slice_column(soap, species):
    
    col_names = []
    species = sorted(species)
    # Iterate over all combinations_with_replacement of your species
    # This must match the (i,j) ordering that SOAP itself uses under the hood.
    for sp1, sp2 in itertools.combinations_with_replacement(species, 2):
        # soap.get_location((sp1, sp2)) returns a slice start:stop for that pair
        sl = soap.get_location((sp1, sp2))
        block_size = sl.stop - sl.start

        # Number each feature in that block from 1..block_size
        for i in range(1, block_size + 1):
            col_names.append(f"{sp1}-{sp2}_{i}")

    return col_names