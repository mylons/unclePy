import h5py
import numpy as np
import pandas as pd
import re
from datetime import datetime
from sqlalchemy import create_engine


class HDF5:
    # TODO insert more assertions for code checking
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

    exp_exists(engine)
        Returns experiment ID if it exists, otherwise returns False

    exp_instrument_exists(engine)
        Returns instrument ID if it exists, otherwise returns False

    exp_product_exists(engine)
        Returns product ID if it exists, otherwise returns False

    write_exp_info_sql(engine, datetime_needed)
        Saves experiment metadata to PostgreSQL database

    write_instrument_info_sql(engine, datetime_needed)
        Saves instrument metadata to PostgreSQL database

    write_product_info_sql(engine)
        Saves product info metadata to PostgreSQL database

    df_to_sql(df, well, engine)
        Returns input dataframe with additional columns added to match
        associated database tables

    wells()
        Returns names of wells used in experiment

    well_name_to_num(well)
        Returns well number converted from input well name
        Example: 'A1' -> 'Well_01'

    samples()
        Returns sample names/descriptions

    """
    def __init__(self, file_path, uncle_experiment_id):
        self.file = h5py.File(file_path, 'r')
        self.uncle_experiment_id = uncle_experiment_id

    def exp_name(self):
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

    def exp_date(self):
        """
        Returns
        -------
        pd.Timestamp
            Date of experiment
        """
        date = self.exp_name().split('-')[0]
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
        inst_num = self.exp_name().split('-')[1]
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
        return self.exp_name().split('-')[2]

    def exp_plate_type(self):
        """
        Returns
        -------
        str
            Plate type used in experiment
        """
        plate_info = self.exp_name().split('-')[-1]
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
        plate_info = self.exp_name().split('-')[-1]
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
        plate_info = self.exp_name().split('-')[-1]
        plate_side = re.search(r'\D+$', plate_info).group()
        assert plate_side.lower() in ['l', 'r'],\
            'Incorrect plate side. Plate side should be one of: L, R'
        return plate_side

    def exp_exists(self, engine):
        """
        NOTE: This checks if the name of the experiment already exists
              It does not make any checks on the experiment ID

        Parameters
        ----------
        engine : sqlalchemy Engine
            Passed in from calling function. Engine to connect to database.

        Returns
        -------
        int or False
            int: experiment ID, if it exists
            False: if experiment does not exist
        """
        with engine.connect() as con:
            exp_id = con.execute("SELECT id FROM uncle_experiments "
                                 "WHERE name = '{}';".
                                 format(self.exp_name()))
            exp_id = exp_id.mappings().all()

        if exp_id:
            return exp_id[0]['id']
        else:
            return False

    def exp_instrument_exists(self, engine):
        """
        Parameters
        ----------
        engine : sqlalchemy Engine
            Passed in from calling function. Engine to connect to database.

        Returns
        -------
        int or False
            int: instrument ID, if it exists
            False: if experiment instrument does not exist
        """
        with engine.connect() as con:
            inst_id = con.execute("SELECT id FROM uncle_instruments "
                                  "WHERE id = '{}';".
                                  format(self.exp_inst_num()))
            inst_id = inst_id.mappings().all()

        if inst_id:
            return inst_id[0]['id']
        else:
            return False

    def exp_product_exists(self, engine):
        """
        Parameters
        ----------
        engine : sqlalchemy Engine
            Passed in from calling function. Engine to connect to database.

        Returns
        -------
        int or False
            int: product ID, if it exists
            False: if product does not exist
        """
        with engine.connect() as con:
            prod_id = con.execute("SELECT id FROM products "
                                  "WHERE name = '{}';".
                                  format(self.exp_product()))
            prod_id = prod_id.mappings().all()

        if prod_id:
            return prod_id[0]['id']
        else:
            return False

    def write_exp_info_sql(self, engine, datetime_needed = True):
        """
        Parameters
        ----------
        engine : sqlalchemy Engine
            Passed in from calling function. Engine to connect to database.
        datetime_needed : bool (default = True)
            Whether to insert "created_at", "updated_at" columns
            These are necessary for Rails tables

        Returns
        -------
        None
        """
        with engine.connect() as con:
            exp_id = con.execute("SELECT id FROM uncle_experiments "
                                 "WHERE name = '{}';".
                                 format(self.exp_name()))
            exp_id = exp_id.mappings().all()
        if exp_id:
            return

        if self.exp_instrument_exists(engine):
            inst_id = self.exp_instrument_exists(engine)
        # Write instrument info if it does not exist
        elif not self.exp_instrument_exists(engine):
            self.write_instrument_info_sql(engine)
            inst_id = self.exp_instrument_exists(engine)

        if self.exp_product_exists(engine):
            prod_id = self.exp_product_exists(engine)
        # Write product info if it does not exist
        elif not self.exp_product_exists(engine):
            self.write_product_info_sql(engine)
            prod_id = self.exp_product_exists(engine)

        exp_info = {'name': [self.exp_name()],
                    'date': [self.exp_date()],
                    'uncle_instrument_id': inst_id,
                    'product_id': prod_id,
                    'exp_type': [self.exp_plate_type()],
                    'plate_generation': [self.exp_generation()],
                    'plate_side': [self.exp_plate_side()]}
        df = pd.DataFrame(exp_info)
        if datetime_needed:
            df = add_datetime(df)
        df.to_sql('uncle_experiments', engine, if_exists = 'append',
                  index = False)

    def write_instrument_info_sql(self, engine, datetime_needed = True):
        """
        Parameters
        ----------
        engine : sqlalchemy Engine
            Passed in from calling function. Engine to connect to database.
        datetime_needed : bool (default = True)
            Whether to insert "created_at", "updated_at" columns
            These are necessary for Rails tables

        Returns
        -------
        None
        """
        with engine.connect() as con:
            inst_id = con.execute("SELECT id FROM uncle_instruments "
                                  "WHERE id = '{}';".
                                  format(self.exp_inst_num()))
            inst_id = inst_id.mappings().all()

        if inst_id:
            return

        # TODO complete this with real info
        inst_info = {'id': [int(self.exp_inst_num())],
                     'name': ['Uncle_01'],
                     'location': ['Shnider/Hough lab'],
                     'model': ['Uncle']}
        df = pd.DataFrame(inst_info)
        if datetime_needed:
            df = add_datetime(df)
        df.to_sql('uncle_instruments', engine, if_exists = 'append',
                  index = False)

    def write_product_info_sql(self, engine):
        """
        Parameters
        ----------
        engine : sqlalchemy Engine
            Passed in from calling function. Engine to connect to database.

        Returns
        -------
        None
        """
        with engine.connect() as con:
            prod_id = con.execute("SELECT id FROM products "
                                  "WHERE name = '{}';".
                                  format(self.exp_product()))
            prod_id = prod_id.mappings().all()

        if prod_id:
            return

        # TODO will these have catalog numbers?
        prod_info = {'name': [self.exp_product()],
                     'active': 'true'}
        df = pd.DataFrame(prod_info)
        df.to_sql('products', engine, if_exists = 'append', index = False)

    def df_to_sql(self, df, well = None, engine = None):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to be modified for PostgreSQL data table
        well : str
            Single well name, e.g. 'A1'
        engine : sqlalchemy Engine
            Passed in from calling function. Engine to connect to database.

        Returns
        -------
        pd.DataFrame
            Modified to fit database structure
        """
        df['export_type'] = df.name.split('_')[0]
        df = add_datetime(df)
        if well:
            df['well'] = well

        if len(df.name.split('_')) == 2:
            df['dls_data_type'] = df.name.split('_')[1]

        if engine and self.exp_exists(engine):
            df['uncle_experiment_id'] = self.exp_exists(engine)
        # Write experimental info if it does not exist
        elif engine and not self.exp_exists(engine):
            self.write_exp_info_sql(engine)
            df['uncle_experiment_id'] = self.exp_exists(engine)

        return df

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


h1 = HDF5('/Users/jmiller/Desktop/UNcle Files/uni files/210602-01-Seq1 Cas9-pH003R.uni')
h2 = HDF5('/Users/jmiller/Desktop/UNcle Files/uni files/Gen6 uni 1,2,3.uni')
save_path = '/Users/jmiller/Desktop/UNcle Files/Misc/uncle_out.xlsx'
engine = create_engine('postgresql://{}:{}@{}:5432/{}'.format('postgres', '', 'localhost', 'ebase_dev'))

"""
Gen6 1,2,3 = 210607-01-T4 RNA Ligase-Gen006L
Gen6 4,5,6 = 210607-01-T4 RNA Ligase – Gen006R
pH 1,2,3 = 210608-01-T4 RNA Ligase-pH003L
pH 4,5 = 210608-01-T4 RNA Ligase-pH003R
"""
