import h5py
import numpy as np
import pandas as pd
import re
from datetime import datetime
from sqlalchemy import create_engine


class HDF5:
    # TODO either add more "examples" or remove some - make more consistent

    """
    Generic class to load .uni (HDF5) files

    Assumed naming convention:
        Date-InstNum-Prod-PlateInfo.uni
            Date: YYMMDD (e.g. 210602)
            InstNum: Instrument number (e.g. 01)
            Prod: Product name (e.g. Seq1 Cas9)
            PlateInfo: Plate type, generation, side (e.g. pH003R)
            Example: 210602-01-Seq1 Cas9-pH003R.uni

    Attributes
    ----------
    file_path : str
        .uni (HDF5) file to load
    uncle_experiment_id : int
        Database ID for UNcle experiment
    well_set_id : int
        Database ID for associated well set

    Methods
    -------
    exp_set_id()
        Returns database ID for associated experiment set

    exp_file_name()
        Returns name of file associated with experiment

    exp_name
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

    samples()
        Returns sample names/descriptions

    exp_exists()
        Returns experiment ID if it exists, otherwise returns False

    exp_confirm_created()
        Checks experiment has been saved to database. Returns nothing.

    exp_instrument_exists()
        Returns instrument ID if it exists, otherwise returns False

    exp_product_exists()
        Returns product ID if it exists, otherwise returns False

    write_exp_info_sql(datetime_needed)
        Saves experiment metadata to PostgreSQL database

    write_instrument_info_sql(datetime_needed)
        Saves instrument metadata to PostgreSQL database

    write_product_info_sql()
        Saves product info metadata to PostgreSQL database

    df_to_sql(df, well)
        Returns input dataframe with additional columns added to match
        associated database tables

    well_name_to_num(well)
        Returns well number converted from input well name
        Example: 'A1' -> 'Well_01'

    well_name_to_id(well)
        Returns database well ID for input well name

    well_name_to_summary(well)
        Returns database summary ID for input well name
    """
    def __init__(self, file_path, uncle_experiment_id, well_set_id):
        self.file = h5py.File(file_path, 'r')
        self.uncle_experiment_id = uncle_experiment_id
        self.well_set_id = well_set_id
        self.engine = create_engine('postgresql://{}:{}@{}:5432/{}'.
                                    format('postgres',
                                           '',
                                           'localhost',
                                           'ebase_dev'))
        self.mapping_L = {'A1': 'A1', 'B1': 'B1', 'C1': 'C1', 'D1': 'D1',
                          'E1': 'E1', 'F1': 'F1', 'G1': 'G1', 'H1': 'H1',
                          'I1': 'A2', 'J1': 'B2', 'K1': 'C2', 'L1': 'D2',
                          'M1': 'E2', 'N1': 'F2', 'O1': 'G2', 'P1': 'H2',
                          'A2': 'A3', 'B2': 'B3', 'C2': 'C3', 'D2': 'D3',
                          'E2': 'E3', 'F2': 'F3', 'G2': 'G3', 'H2': 'H3',
                          'I2': 'A4', 'J2': 'B4', 'K2': 'C4', 'L2': 'D4',
                          'M2': 'E4', 'N2': 'F4', 'O2': 'G4', 'P2': 'H4',
                          'A3': 'A5', 'B3': 'B5', 'C3': 'C5', 'D3': 'D5',
                          'E3': 'E5', 'F3': 'F5', 'G3': 'G5', 'H3': 'H5',
                          'I3': 'A6', 'J3': 'B6', 'K3': 'C6', 'L3': 'D6',
                          'M3': 'E6', 'N3': 'F6', 'O3': 'G6', 'P3': 'H6'}
        self.mapping_R = {'A1': 'A7', 'B1': 'B7', 'C1': 'C7', 'D1': 'D7',
                          'E1': 'E7', 'F1': 'F7', 'G1': 'G7', 'H1': 'H7',
                          'I1': 'A8', 'J1': 'B8', 'K1': 'C8', 'L1': 'D8',
                          'M1': 'E8', 'N1': 'F8', 'O1': 'G8', 'P1': 'H8',
                          'A2': 'A9', 'B2': 'B9', 'C2': 'C9', 'D2': 'D9',
                          'E2': 'E9', 'F2': 'F9', 'G2': 'G9', 'H2': 'H9',
                          'I2': 'A10', 'J2': 'B10', 'K2': 'C10', 'L2': 'D10',
                          'M2': 'E10', 'N2': 'F10', 'O2': 'G10', 'P2': 'H10',
                          'A3': 'A11', 'B3': 'B11', 'C3': 'C11', 'D3': 'D11',
                          'E3': 'E11', 'F3': 'F11', 'G3': 'G11', 'H3': 'H11',
                          'I3': 'A12', 'J3': 'B12', 'K3': 'C12', 'L3': 'D12',
                          'M3': 'E12', 'N3': 'F12', 'O3': 'G12', 'P3': 'H12'}

    # ----------------------------------------------------------------------- #
    # EXPERIMENT METADATA                                                     #
    # ----------------------------------------------------------------------- #
    def exp_set_id(self):
        """
        Returns
        -------
        int
            Database ID for associated experiment set
        """
        with self.engine.connect() as con:
            exp_set_id = con.execute("SELECT uncle_experiment_set_id "
                                     "FROM uncle_experiments "
                                     "WHERE id = {};".
                                     format(self.uncle_experiment_id))
            exp_set_id = exp_set_id.mappings().all()
        return exp_set_id[0]['uncle_experiment_set_id']

    def exp_file_name(self):
        """
        Returns
        -------
        str
            Name of current run
        """
        run_name = self.file['Application1']['Run1'].attrs['Run Name'].\
            decode('utf-8')
        assert run_name.count('-') == 3,\
            'Incorrect file name format. Format should be: ' \
            'Date (YYMMDD) – Instrument # – Protein Name – ' \
            'Plate Type/Generation/Side'
        return run_name

    def exp_name(self):
        """
        Returns
        -------
        str
            Name of experiment - same as file name without plate side
        """
        return self.exp_file_name()[:-1]

    def exp_date(self):
        """
        Returns
        -------
        pd.Timestamp
            Date of experiment
        """
        date = self.exp_file_name().split('-')[0]
        assert date.isnumeric(),\
            'Incorrect date format. Format should be: YYMMDD'
        assert len(date) == 6,\
            'Incorrect date format. Format should be: YYMMDD'
        return pd.to_datetime(date, yearfirst = True)

    def exp_inst_num(self):
        """
        Returns
        -------
        int
            Instrument number used in experiment
        """
        inst_num = self.exp_file_name().split('-')[1]
        assert inst_num.isnumeric(),\
            'Incorrect instrument number format. Format should be: ##'
        return int(inst_num)

    def exp_product(self):
        """
        Returns
        -------
        str
            Product used in experiment
        """
        return self.exp_file_name().split('-')[2]

    def exp_plate_type(self):
        """
        Returns
        -------
        str
            Plate type used in experiment
        """
        plate_info = self.exp_file_name().split('-')[-1]
        plate_type = re.search(r'\D+', plate_info).group()
        assert plate_type.lower() in ['ph', 'cond', 'gen'],\
            'Incorrect plate type. Plate type should be one of: pH, cond, gen'
        return plate_type

    def exp_generation(self):
        """
        Returns
        -------
        str
            Generation of plate layout used in experiment
        """
        plate_info = self.exp_file_name().split('-')[-1]
        plate_gen = re.search(r'\d+', plate_info).group()
        assert plate_gen.isnumeric(),\
            'Incorrect experiment generation format. Format should be: ###'
        return plate_gen

    def exp_plate_side(self):
        """
        Returns
        -------
        str
            Plate side used in experiment
        """
        plate_info = self.exp_file_name().split('-')[-1]
        plate_side = re.search(r'\D+$', plate_info).group()
        assert plate_side.lower() in ['l', 'r'],\
            'Incorrect plate side. Plate side should be one of: L, R'
        return plate_side

    # ----------------------------------------------------------------------- #
    # EXPERIMENT INFORMATION                                                  #
    # ----------------------------------------------------------------------- #
    def wells(self):
        """
         {uni well name: actual well name}

        Returns
        -------
        np.array
            Well names

        Examples
        --------
        np.array(['A1', 'B1', ...])
        """
        if self.exp_plate_side() == 'L':
            mapping = self.mapping_L
        elif self.exp_plate_side() == 'R':
            mapping = self.mapping_R
        else:
            raise AttributeError('Cannot determine plate side for well '
                                 'mapping.')

        wells = []
        for i in self.file['Application1']['Run1']['SampleData']:
            well = i[0].decode('utf-8')
            wells = np.append(wells, mapping[well])
        return wells

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
        if self.exp_plate_side() == 'L':
            mapping = self.mapping_L
        elif self.exp_plate_side() == 'R':
            mapping = self.mapping_R
        else:
            raise AttributeError('Cannot determine plate side for well '
                                 'mapping.')
        samples = []

        for i in self.file['Application1']['Run1']['SampleData']:
            sample_name = i[1].decode('utf-8').split(' ')
            sample = sample_name[-1]
            corrected_sample_name = [mapping[sample] if i == sample
                                     else i for i in sample_name]
            samples = np.append(samples, ' '.join(corrected_sample_name))
        return samples

    # ----------------------------------------------------------------------- #
    # EXPERIMENT CHECKS                                                       #
    # ----------------------------------------------------------------------- #
    def exp_exists(self):
        # TODO may need to look at more than just name e.g. plate side

        """
        NOTE: This checks if the name of the experiment already exists
              It does not make any checks on the experiment ID

        Returns
        -------
        int or False
            int: experiment ID, if it exists
            False: if experiment does not exist
        """
        with self.engine.connect() as con:
            exp_id = con.execute("SELECT id "
                                 "FROM uncle_experiments "
                                 "WHERE uncle_experiment_set_id = '{}' "
                                 "AND plate_side = '{}';".format(
                                     self.exp_set_id(),
                                     self.exp_plate_side()))
            exp_id = exp_id.mappings().all()

        if exp_id:
            return exp_id[0]['id']
        else:
            return False

    def exp_confirm_created(self):
        """
        Returns
        -------
        None
        """
        with self.engine.connect() as con:
            exp = con.execute("SELECT id FROM uncle_experiments "
                              "WHERE id = '{}';".
                              format(self.uncle_experiment_id))
            exp = exp.mappings().all()
        assert exp, 'Could not find UNcle experiment. ' \
                    'Confirm experiment has been created.'

    def exp_instrument_exists(self):
        """
        Returns
        -------
        int or False
            int: instrument ID, if it exists
            False: if experiment instrument does not exist
        """
        with self.engine.connect() as con:
            inst_id = con.execute("SELECT id FROM uncle_instruments "
                                  "WHERE id = '{}';".
                                  format(self.exp_inst_num()))
            inst_id = inst_id.mappings().all()

        if inst_id:
            return inst_id[0]['id']
        else:
            return False

    def exp_product_exists(self):
        """
        Returns
        -------
        int or False
            int: product ID, if it exists
            False: if product does not exist
        """
        with self.engine.connect() as con:
            prod_id = con.execute("SELECT id FROM products "
                                  "WHERE name = '{}';".
                                  format(self.exp_product()))
            prod_id = prod_id.mappings().all()

        if prod_id:
            return prod_id[0]['id']
        else:
            return False

    # ----------------------------------------------------------------------- #
    # WRITE DATA TO POSTGRESQL                                                #
    # ----------------------------------------------------------------------- #
    def write_exp_set_info_sql(self, datetime_needed = True):
        """
        Parameters
        ----------
        datetime_needed : bool (default = True)
            Whether to insert "created_at", "updated_at" columns
            These are necessary for Rails tables

        Returns
        -------
        None
        """
        if self.exp_product_exists():
            product_id = self.exp_product_exists()
        # Write product info if it does not exist
        else:
            self.write_product_info_sql()
            product_id = self.exp_product_exists()

        exp_set_info = {'name': [self.exp_name()],
                        'product_id': product_id,
                        'exp_type': [self.exp_plate_type()],
                        'plate_generation': [self.exp_generation()]}
        df = pd.DataFrame(exp_set_info)
        if datetime_needed:
            df = add_datetime(df)
        update_params = df.to_dict('records')[0]

        with self.engine.connect() as con:
            con.execute("UPDATE uncle_experiment_sets SET "
                        "name = '{}', "
                        "product_id = '{}', "
                        "exp_type = '{}', "
                        "plate_generation = '{}', "
                        "created_at = '{}', "
                        "updated_at = '{}' "
                        "WHERE id = {};".format(
                            update_params['name'],
                            update_params['product_id'],
                            update_params['exp_type'],
                            update_params['plate_generation'],
                            update_params['created_at'],
                            update_params['updated_at'],
                            self.exp_set_id()))

    def write_exp_info_sql(self, datetime_needed = True):
        """
        Parameters
        ----------
        datetime_needed : bool (default = True)
            Whether to insert "created_at", "updated_at" columns
            These are necessary for Rails tables

        Returns
        -------
        None
        """
        if self.exp_exists():
            return

        if self.exp_instrument_exists():
            inst_id = self.exp_instrument_exists()
        # Write instrument info if it does not exist
        else:
            self.write_instrument_info_sql()
            inst_id = self.exp_instrument_exists()

        exp_info = {'uncle_experiment_set_id': self.exp_set_id(),
                    'uncle_instrument_id': inst_id,
                    'plate_side': [self.exp_plate_side()],
                    'date': [self.exp_date()]}
        df = pd.DataFrame(exp_info)
        if datetime_needed:
            df = add_datetime(df)
        update_params = df.to_dict('records')[0]

        with self.engine.connect() as con:
            con.execute("UPDATE uncle_experiments SET "
                        "uncle_experiment_set_id = {},"
                        "uncle_instrument_id = '{}',"
                        "plate_side = '{}',"
                        "date = '{}',"
                        "created_at = '{}',"
                        "updated_at = '{}'"
                        "WHERE id = {};".format(
                            update_params['uncle_experiment_set_id'],
                            update_params['uncle_instrument_id'],
                            update_params['plate_side'],
                            update_params['date'],
                            update_params['created_at'],
                            update_params['updated_at'],
                            self.uncle_experiment_id
                        ))

    def write_instrument_info_sql(self, datetime_needed = True):
        """
        Parameters
        ----------
        datetime_needed : bool (default = True)
            Whether to insert "created_at", "updated_at" columns
            These are necessary for Rails tables

        Returns
        -------
        None
        """
        with self.engine.connect() as con:
            inst_id = con.execute("SELECT id FROM uncle_instruments "
                                  "WHERE id = '{}';".
                                  format(self.exp_inst_num()))
            inst_id = inst_id.mappings().all()

        if inst_id:
            return

        inst_info = {'id': [int(self.exp_inst_num())],
                     'name': ['UNcle_01'],
                     'location': ['Shnider/Hough lab'],
                     'model': ['UNcle']}
        df = pd.DataFrame(inst_info)
        if datetime_needed:
            df = add_datetime(df)
        df.to_sql('uncle_instruments', self.engine, if_exists = 'append',
                  index = False)

    def write_product_info_sql(self):
        """
        Returns
        -------
        None
        """
        if self.exp_product_exists():
            return

        prod_info = {'name': [self.exp_product()],
                     'active': 'true'}
        df = pd.DataFrame(prod_info)
        df.to_sql('products', self.engine, if_exists = 'append', index = False)

    def df_to_sql(self, df, well = None):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to be modified for PostgreSQL data table
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        pd.DataFrame
            Modified to fit database structure
        """
        df = add_datetime(df)
        if well:
            df['well'] = well

        if df.name == 'summary':
            if self.exp_exists():
                df['uncle_experiment_id'] = self.exp_exists()
            # Write experimental info if it does not exist
            else:
                self.write_exp_info_sql()
                df['uncle_experiment_id'] = self.exp_exists()

        return df

    # ----------------------------------------------------------------------- #
    # UTILITY FUNCTIONS                                                       #
    # ----------------------------------------------------------------------- #
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

    def well_name_to_id(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        int
            Database well_id
        """
        with self.engine.connect() as con:
            well_id = con.execute("SELECT well_id FROM well_set_wells "
                                  "WHERE well_set_id = {} "
                                  "AND plate_address = '{}';".
                                  format(self.well_set_id, well))
            well_id = well_id.mappings().all()
        return well_id[0]['well_id']

    def well_name_to_summary(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        int
            Database summary_id
        """
        if well[0].isalpha():
            well = self.well_name_to_id(well)
        with self.engine.connect() as con:
            summary_id = con.execute("SELECT id FROM uncle_dls_summary "
                                     "WHERE well_id = {};".format(well))
            summary_id = summary_id.mappings().all()
        return summary_id[0]['id']


def add_datetime(df):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to add created_at, updated_at columns to

    Returns
    -------
    pd.DataFrame
        Input dataframe with created_at, updated_at columns added
    """
    dt = datetime.now()
    df['created_at'] = dt
    df['updated_at'] = dt
    return df


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


"""
h1 = HDF5('/Users/jmiller/Desktop/UNcle Files/uni files/210602-01-Seq1 Cas9-pH003R.uni', 71)
h2 = HDF5('/Users/jmiller/Desktop/UNcle Files/uni files/Gen6 uni 1,2,3.uni', 72)
save_path = '/Users/jmiller/Desktop/UNcle Files/Misc/uncle_out.xlsx'
engine = create_engine('postgresql://{}:{}@{}:5432/{}'.format('postgres', '', 'localhost', 'ebase_dev'))

Gen6 1,2,3 = 210607-01-T4 RNA Ligase-Gen006L
Gen6 4,5,6 = 210607-01-T4 RNA Ligase – Gen006R
pH 1,2,3 = 210608-01-T4 RNA Ligase-pH003L
pH 4,5 = 210608-01-T4 RNA Ligase-pH003R
"""
