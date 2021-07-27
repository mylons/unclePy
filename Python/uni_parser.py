from hdf5 import HDF5
from sls import SLS
from dls import DLS
import argparse

parser = argparse.ArgumentParser(description = 'Parse .uni file')
parser.add_argument('uni_file',
                    help = 'Path to .uni file',
                    type = str)
parser.add_argument('uncle_experiment_id',
                    help = 'Database ID for UNcle experiment',
                    type = int)
parser.add_argument('well_set_id',
                    help = 'Database ID for associated well set',
                    type = int)

args = parser.parse_args()

if __name__ == '__main__':
    hdf = HDF5(args.uni_file, args.uncle_experiment_id, args.well_set_id)
    dls = DLS(args.uni_file, args.uncle_experiment_id, args.well_set_id)
    sls = SLS(args.uni_file, args.uncle_experiment_id, args.well_set_id)

    hdf.write_exp_set_info_sql()
    dls.write_dls_summary_sql()
    dls.write_dls_correlation_sql()
    dls.write_dls_intensity_sql()
    dls.write_dls_mass_sql()
    sls.write_sls_summary_sql()
    sls.write_sls_266_sql()
    sls.write_sls_473_sql()
    sls.write_sls_bcm_sql()
