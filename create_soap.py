from dscribe.descriptors import SOAP
import numpy as np

def soap():
    # Define SOAP descriptor
    soap = SOAP(
        species=species,  # Elements in Methane
        periodic=False,
        r_cut=3.0,           # Cutoff radius (Å)
        n_max=n_max,             # Radial basis functions
        l_max=l_max,             # Angular channels
        sigma=0.5            # Gaussian width
    )
    
    # Generate SOAP descriptors
    soap_methane = soap.create(methane)
    
    return soap_methane
