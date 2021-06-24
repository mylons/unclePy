import h5py
import numpy as np
import re


class HDF5:
    """
    Generic class to load .uni (HDF5) files

    Attributes
    ----------
    file_path : str
        .uni (HDF5) file to load

    Methods
    -------
    run_name()
        Returns name of experimental run

    exp_date()
        Returns date of experiment

    exp_inst_num()
        Returns number of instrument used in experiment

    exp_product()
        Returns product tested in experiment

    exp_plate_type()
        Returns type of plate/screen used in experiment (pH, cond, gen)

    exp_generation()
        Returns generation of plate layout used in experiment

    exp_plate_side()
        Returns side of plate used in experiment (L/R)

    wells()
        Returns names of wells used in experiment

    well_name_to_num(well)
        Returns well number converted from input well name
        Example: 'A1' -> 'Well_01'

    samples()
        Returns sample names/descriptions

    """
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')

    # ----------------------------------------------------------------------- #
    # GENERIC DATA COLLECTION                                                 #
    # ----------------------------------------------------------------------- #
    def run_name(self):
        """
        Returns
        -------
        str
            Name of current run
        """
        run_name = self.file['Application1']['Run1'].attrs['Run Name'].\
            decode('utf-8')
        return run_name

    def exp_date(self):
        """
        Returns
        -------
        str
            Date of experiment (YYMMDD)
        """
        return self.run_name().split('-')[0]

    def exp_inst_num(self):
        """
        Returns
        -------
        str
            Instrument number used in experiment
        """
        return self.run_name().split('-')[1]

    def exp_product(self):
        """
        Returns
        -------
        str
            Product used in experiment
        """
        return self.run_name().split('-')[2]

    def exp_plate_type(self):
        """
        Returns
        -------
        str
            Plate type used in experiment
        """
        plate_info = self.run_name().split('-')[-1]
        plate_type = re.search(r'\D+', plate_info)
        return plate_type.group()

    def exp_generation(self):
        """
        Returns
        -------
        str
            Generation of plate layout used in experiment
        """
        plate_info = self.run_name().split('-')[-1]
        plate_gen = re.search(r'\d+', plate_info)
        return plate_gen.group()

    def exp_plate_side(self):
        """
        Returns
        -------
        str
            Plate side used in experiment
        """
        plate_info = self.run_name().split('-')[-1]
        plate_side = re.search(r'\D+$', plate_info)
        return plate_side.group()

    def wells(self):
        """
        Returns
        -------
        np.array
            Well names

        Examples
        --------
        np.array(['A1', 'B1', ...])
        """
        wells = []
        for i in self.file['Application1']['Run1']['SampleData']:
            wells = np.append(wells, i[0].decode('utf-8'))
        return wells

    def well_name_to_num(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        string
            Well number, e.g. 'Well_01'
        """
        well_num = np.argwhere(self.wells() == well)[0][0] + 1
        well_num = f'Well_{well_num:02}'
        return well_num

    def samples(self):
        """
        Returns
        -------
        np.array
            Sample names

        Examples
        --------
        np.array(['0.1 mg/ml Uni A1', '0.1 mg/ml Uni B1', ...])
        """
        samples = []
        for i in self.file['Application1']['Run1']['SampleData']:
            samples = np.append(samples, i[1].decode('utf-8'))
        return samples


def verify(value):
    """
    Parameters
    ----------
    value : int, float
        Any value to verify is legitimate

    Returns
    -------
    int, float, np.nan
        Depends on input value
        Returns input value if valid, otherwise np.nan

    """
    if value != -1:
        return value
    else:
        return np.nan


h1 = HDF5('/Users/jmiller/Desktop/UNcle Files/uni files/210602-01-Seq1 Cas9-pH003R.uni')
h2 = HDF5('/Users/jmiller/Desktop/UNcle Files/uni files/Gen6 uni 1,2,3.uni')
save_path = '/Users/jmiller/Desktop/UNcle Files/Misc/uncle_out.xlsx'

"""
Gen6 1,2,3 = 210607-01-T4 RNA Ligase-Gen006L
Gen6 4,5,6 = 210607-01-T4 RNA Ligase – Gen006R
pH 1,2,3 = 210608-01-T4 RNA Ligase-pH003L
pH 4,5 = 210608-01-T4 RNA Ligase-pH003R
"""
