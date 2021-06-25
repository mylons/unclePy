from hdf5 import HDF5, verify, add_datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


class SLS(HDF5):
    """
    Subclass of HDF5 class, specific for SLS portion of experiment

    Attributes
    ----------
    Reference HDF5 class

    Methods
    -------
    sls_temperatures(well)
        Returns temperatures used for single well

    sls_times(well)
        Returns times used for single well

    sls_wavelengths(well)
        Returns wavelengths used for single well

    sls_intensity(well, temp)
        Returns intensities found at a single temperature for single well

    sls_color(well)
        *** NOT CURRENTLY IMPLEMENTED ***
        Returns color for single well

    sls_tms(well)
        Returns all TM values for single well

    sls_tonset(well)
        Returns T_onset value for single well

    sls_tagg266(well)
        Returns T_agg 266 for single well

    sls_tagg473(well)
        Returns T_agg 473 for single well

    sls_bcm(well)
        Returns BCM/nm at all temperatures for single well

    sls_266(well)
        Returns SLS 266 nm/Count at all temperatures for single well

    sls_473(well)
        Returns SLS 473 nm/Count at all temperatures for single well

    sls_spec_well(well)
        Returns pd.DataFrame of intensities for single well

    sls_sum()
        Returns pd.DataFrame of summary for entire experiment

    sls_export(well)
        Returns pd.DataFrame of BCM/nm, SLS 266 nm/Count, SLS 473 nm/Count
        for single well

    write_sls_spec_excel(save_path)
        Saves spectra file (intensity per wavelength at a temperature) to .xlsx

    write_sls_sum_excel(save_path)
        Saves summary file to .xlsx

    write_sls_sum_csv(save_path)
        Saves summary file to .csv

    write_sls_export_excel(save_path)
        Saves BCM/nm, SLS 266 nm/Count, SLS 473 nm/Count (at temperature) file
        to .xlsx

    write_sls_bundle_csv(save_path)
        Saves BCM/nm, SLS 266 nm/Count, SLS 473 nm/Count (at temperature) file
        to .csv

    write_sls_sum_sql(username, password, host, database, datetime_needed)
        Saves summary data to PostgreSQL database

    write_sls_bundle_sql(username, password, host, database, datetime_needed)
        Saves BCM/nm, SLS 266 nm/Count, SLS 473 nm/Count (at temperature) data
        to PostgreSQL database
    """

    # ----------------------------------------------------------------------- #
    # GENERIC DATA COLLECTION                                                 #
    # ----------------------------------------------------------------------- #
    def sls_temperatures(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            Temperatures used in SLS analysis for single well
        """
        well_num = self.well_name_to_num(well)
        meas_dir = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['CorrectedSpectra']

        temps = []
        for i in meas_dir:
            temps = np.append(temps,
                              meas_dir[i].attrs['Actual temperature'])
        return temps

    def sls_times(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            Times used in SLS analysis for single well
        """
        well_num = self.well_name_to_num(well)
        meas_dir = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['CorrectedSpectra']

        times = []
        for i in meas_dir:
            times = np.append(times,
                              meas_dir[i].attrs['Actual Time'])
        return times

    # ----------------------------------------------------------------------- #
    # DATA COLLECTION FOR SLS SPEC                                            #
    # ----------------------------------------------------------------------- #
    def sls_wavelengths(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            Wavelengths used in SLS analysis for single well
        """
        well_num = self.well_name_to_num(well)
        wavelengths = self.file['Application1']['Run1'][well_num] \
                          ['Fluor_SLS_Data']['CorrectedSpectra']['0001'][:, 0]
        return wavelengths

    def sls_intensity(self, well, temp):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'
        temp : float
            Single temperature value from file
            Does a direct lookup so value should be taken directly from file

        Returns
        -------
        np.array
            Intensities for a single temp of single well
        """
        well_num = self.well_name_to_num(well)
        temps = self.sls_temperatures(well)
        # Cannot index into HDF5 dataset, therefore need to call with dict key
        index = np.flatnonzero(temps == temp)[0]
        inten_cnt = f'{index + 1:04}'

        inten_meas = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['CorrectedSpectra'][inten_cnt]

        # Assert we are looking at correct well, temp, etc.
        np.testing.assert_array_equal(inten_meas[:, 0],
                                      self.sls_wavelengths(well))

        return inten_meas[:, 1]

    # ----------------------------------------------------------------------- #
    # DATA COLLECTION FOR SLS SUMMARY                                         #
    # ----------------------------------------------------------------------- #
    def sls_color(self, well):
        """
        TODO: color is currently blank for all files. Is there ever a value?

        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        None (because currently not used)
        """
        return np.nan

    def sls_tms(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            All TM values for single well

        Examples
        --------
            np.array([42.48, 82.4 ,  0.  ,  0.  ])
        """
        well_num = self.well_name_to_num(well)
        tms = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['Tms'][0]
        return tms

    def sls_tonset(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        float
            Tonset value for single well
        """
        well_num = self.well_name_to_num(well)
        tonset = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['TonsetBCM'][0]
        tonset = verify(tonset)
        return tonset

    def sls_tagg266(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        float
            Tagg266 for single well
        """
        well_num = self.well_name_to_num(well)
        tagg266 = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['Tagg266'][0]
        tagg266 = verify(tagg266)
        return tagg266

    def sls_tagg473(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        float
            Tagg473 for single well
        """
        well_num = self.well_name_to_num(well)
        tagg473 = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['Tagg473'][0]
        tagg473 = verify(tagg473)
        return tagg473

    # ----------------------------------------------------------------------- #
    # DATA COLLECTION FOR SLS EXPORT                                          #
    # ----------------------------------------------------------------------- #
    def sls_bcm(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            BCM/nm for single well
        """
        well_num = self.well_name_to_num(well)
        bcm = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['BCM'][:]
        return bcm

    def sls_266(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            SLS 266 nm/Count for single well
        """
        well_num = self.well_name_to_num(well)
        sls_266 = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['SLS266'][:]
        return sls_266

    def sls_473(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            SLS 473 nm/Count for single well
        """
        well_num = self.well_name_to_num(well)
        sls_473 = self.file['Application1']['Run1'][well_num] \
            ['Fluor_SLS_Data']['Analysis']['SLS473'][:]
        return sls_473

    # ----------------------------------------------------------------------- #
    # DATAFRAME ASSEMBLY                                                      #
    # ----------------------------------------------------------------------- #
    def sls_spec_well(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        pd.DataFrame
            Full dataframe of SLS intensities for single well
            This is comparable to an Excel tab for one well, e.g. 'A1'
        """
        temps = self.sls_temperatures(well)
        times = self.sls_times(well)
        waves = self.sls_wavelengths(well)
        inten = [pd.Series(self.sls_intensity(well, temp)) for temp in temps]

        cols = [f'Temp :{temps[i]:02f}, Time:{times[i]:.1f}'
                for i in range(len(temps))]  # formatting matches file

        # Transpose due to the way dataframe is compiled
        df = pd.DataFrame(data = inten).T
        df.columns = cols
        df.index = waves
        df.index.name = 'Wavelength'

        return df

    def sls_sum(self):
        """
        Returns
        -------
        pd.DataFrame
            Full dataframe for SLS Summary
        """
        wells = self.wells()
        samples = self.samples()

        cols = ['color', 'well', 'sample', 't_onset', 't_agg_266', 't_agg_473']
        # Determine how many Tm columns
        tm_cols = []
        for i in self.wells():
            tm_cols = np.append(tm_cols, np.flatnonzero(self.sls_tms(i)))
        max_tm = int(np.max(tm_cols) + 1)
        cols.extend(['t_m_{}'.format(i + 1) for i in range(max_tm)])

        df = pd.DataFrame(columns = cols)

        for i in wells:
            well_sum = {'color': self.sls_color(i),
                        'well': i,
                        't_onset': self.sls_tonset(i),
                        't_agg_266': self.sls_tagg266(i),
                        't_agg_473': self.sls_tagg473(i)}

            sample_index = np.flatnonzero(
                np.char.find(self.samples(), i) != -1)
            well_sum['sample'] = samples[sample_index][0]

            tms = self.sls_tms(i)
            for j in range(max_tm):
                well_sum['t_m_{}'.format(j + 1)] = \
                    tms[j] if tms[j] != 0 else np.nan

            df = df.append(well_sum, ignore_index = True)

        return df

    def sls_export(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        pd.DataFrame
            Full dataframe of BCM/nm, SLS 266 nm/Count, SLS 473 nm/Count
                for single well
            This is comparable to an Excel tab for one well, e.g. 'A1'
        """
        temps = self.sls_temperatures(well)
        bcm = self.sls_bcm(well)
        sls_266 = self.sls_266(well)
        sls_473 = self.sls_473(well)

        cols = ['temperature', 'bcm', 'sls_266', 'sls_473']

        df = pd.DataFrame(data = [temps, bcm, sls_266, sls_473]).T
        df.columns = cols

        return df

    def sls_bundle(self, well):
        """
        NOTE: the bundle file is identical to the "SLS export" file so this
              method simply calls self.sls_export()

        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        pd.DataFrame
            Full dataframe of BCM/nm, SLS 266 nm/Count, SLS 473 nm/Count
                for single well
            This is comparable to an Excel tab for one well, e.g. 'A1'
        """
        df = self.sls_export(well)
        return df

    # ----------------------------------------------------------------------- #
    # WRITE DATA TO EXCEL, CSV, SQL                                           #
    # ----------------------------------------------------------------------- #
    def write_sls_spec_excel(self, save_path):
        """
        Parameters
        ----------
        save_path : str
            Directory to save Excel file to

        Returns
        -------
        None
        """
        wells = self.wells()
        with pd.ExcelWriter(save_path) as writer:
            for well in wells:
                df = self.sls_spec_well(well)
                df.to_excel(writer, sheet_name = well)

    def write_sls_sum_excel(self, save_path):
        """
        Parameters
        ----------
        save_path : str
            Directory to save Excel file to

        Returns
        -------
        None
        """
        df = self.sls_sum()
        df.to_excel(save_path, index = False)

    def write_sls_sum_csv(self, save_path):
        """
        Parameters
        ----------
        save_path : str
            Directory to save CSVs to

        Returns
        -------
        None
        """
        df = self.sls_sum()
        run_name = self.exp_name()
        df.to_csv('{}/{}-SLS Sum.csv'.format(save_path, run_name),
                  index = False)

    def write_sls_export_excel(self, save_path):
        """
        Parameters
        ----------
        save_path : str
            Directory to save Excel file to

        Returns
        -------
        None
        """
        wells = self.wells()
        with pd.ExcelWriter(save_path) as writer:
            for well in wells:
                df = self.sls_export(well)
                df.to_excel(writer, sheet_name = well, index = False)

    def write_sls_bundle_csv(self, save_path):
        """
        Parameters
        ----------
        save_path : str
            Directory to save CSVs to

        Returns
        -------
        None
        """
        wells = self.wells()
        run_name = self.exp_name()
        for well in wells:
            df = self.sls_bundle(well)
            df.to_csv('{}/{}-SLS Bundle-{}.csv'.format(save_path,
                                                       run_name, well),
                      index = False)

    # ----------------------------------------------------------------------- #
    # WRITE DATA TO POSTGRESQL                                                #
    # ----------------------------------------------------------------------- #
    def write_sls_sum_sql(self, username, password, host, database,
                          datetime_needed = True):
        """
        Parameters
        ----------
        username : str
            Username for database access (e.g. "postgres")
        password : str
            Password for database access (likely none, i.e. empty string: "")
        host : str
            Host address for database access (e.g. "ebase-db-c")
        database : str
            Database name (e.g. "ebase_dev")
        datetime_needed : bool (default = True)
            Whether to insert "created_at", "updated_at" columns
            These are necessary for Rails tables

        Returns
        -------
        None
        """
        df = self.sls_sum()
        # If uploading raw file, need to append datetime for Rails table
        df['export_type'] = 'summary'
        if datetime_needed:
            df = add_datetime(df)

        engine = create_engine('postgresql://{}:{}@{}:5432/{}'.format(
            username, password, host, database))
        df.to_sql('uncle_sls', engine, if_exists = 'append', index = False)

    def write_sls_bundle_sql(self, username, password, host, database,
                             datetime_needed = True):
        """
        Parameters
        ----------
        username : str
            Username for database access (e.g. "postgres")
        password : str
            Password for database access (likely none, i.e. empty string: "")
        host : str
            Host address for database access (e.g. "ebase-db-c")
        database : str
            Database name (e.g. "ebase_dev")
        datetime_needed : bool (default = True)
            Whether to insert "created_at", "updated_at" columns
            These are necessary for Rails tables

        Returns
        -------
        None
        """
        engine = create_engine('postgresql://{}:{}@{}:5432/{}'.format(
            username, password, host, database))

        wells = self.wells()
        for well in wells:
            df = self.sls_bundle(well)
            df['export_type'] = 'bundle'
            df['well'] = well

            # TODO need to grab experiment ID

            if datetime_needed:
                df = add_datetime(df)
            df.to_sql('uncle_sls', engine, if_exists = 'append', index = False)


h5 = SLS('/Users/jmiller/Desktop/UNcle Files/uni files/210602-01-Seq1 Cas9-pH003R.uni')
h6 = SLS('/Users/jmiller/Desktop/UNcle Files/uni files/Gen6 uni 1,2,3.uni')
save_path = '/Users/jmiller/Desktop/UNcle Files/Misc/'
