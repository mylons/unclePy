from hdf5 import HDF5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import Boltzmann, convert_temperature
from sklearn.metrics import mean_squared_error


class DLS(HDF5):
    """
    Subclass of HDF5 class, specific for DLS portion of experiment

    Attributes
    ----------
    Reference HDF5 class

    Methods
    -------
    dls_correlation(well)
        Returns correlation time and amplitude for single well

    dls_intensity(well)
        Returns hydrodynamic diameter and amplitude for single well

    dls_mass(well)
        Returns mass hydrodynamic diameter and amplitude for single well

    dls_summary_color()
        *** Not currently implemented ***

    dls_summary_temperatures()
        Returns temperatures used for all wells

    dls_summary_z_avg_diam(raw, diam)
        Returns Z-average diameters/radii for all wells

    dls_summary_z_avg_diff_coeff()
        Returns Z-average differential coefficients for all wells

    dls_summary_stdev_diam(raw)
        Returns standard deviation of diameter for all wells

    dls_summary_pdi()
        Returns polydispersity index for all wells

    dls_summary_correlation_values(for_plotting)
        Returns experimental and expected values from correlation data/fit
        for all wells. Optionally returns time points used for plotting.

    dls_summary_residuals()
        Returns residuals of correlation plot fit for all wells.

    dls_summary_rmse(mse)
        Returns (root) mean squared error

    dls_summary_intensity()
        Returns intensities for all wells

    dls_summary_pk_mode_diam(raw, diam)
        Returns modal diameters/radii for all peaks for all wells

    dls_summary_pk_est_mw(raw)
        Returns estimated molecular weights for all peaks for all wells

    dls_summary_pk_poly()
        Returns polydispersity percentage for all peaks for all wells

    dls_summary_pk_mass()
        Returns mass percentage for all peaks for all wells

    dls_summary_data_filter()
        Returns filter used for all wells

    dls_summary_viscosity()
        Returns viscosity for all wells

    dls_summary_ri()
        Returns refractive index for all wells

    dls_atten_perc()
        Returns attenuation percentage for all wells

    dls_laser_perc()
        Returns laser percentage for all wells

    dls_summary_der_intensity()
        Returns derived intensiteis for all wells

    dls_summary_min_pk_area()
        Returns minimum peak area for all wells

    dls_summary_min_rh()
        Returns minimum relative humidity for all wells

    dls_summary()
        Returns pd.DataFrame of summary for entire experiment

    write_dls_summary_sql(username, password, host, database)
        Saves summary data to PostgreSQL database

    write_dls_bundle_sql(username, password, host, database)
        Saves intensity, mass, correlation data per well to PostgreSQL database
    """

    def __init__(self, file_path, uncle_experiment_id, well_set_id):
        super().__init__(file_path, uncle_experiment_id, well_set_id)
        # Hydrodynamic diameter is consistently 2x values found in .uni file
        # Is this possibly because values are radii and export is diameter?
        self.factor = 2

    # ----------------------------------------------------------------------- #
    # DATA COLLECTION FOR DLS BUNDLE                                          #
    # ----------------------------------------------------------------------- #
    def dls_correlation(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        pd.DataFrame
            Correlation time(sec) and amplitude for a single well
        """
        well_num = self.well_name_to_num(well)
        corr = self.file['Application1']['Run1'][well_num]['DLS_Data'] \
            ['DLS0001']['ExperimentAveraged']['AverageCorrelation'] \
            ['Correlations'][:]
        # Swap columns to align with typical export
        corr = corr[:, [1, 0]]
        df = pd.DataFrame(corr, columns = ['time', 'amplitude'])
        df['uncle_dls_summary_id'] = self.well_id_to_summary(well)
        return df

    def dls_intensity(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        pd.DataFrame
            Intensity hydrodynamic diameter(nm) and amplitude for a single well
        """
        well_num = self.well_name_to_num(well)
        inten = self.file['Application1']['Run1'][well_num]['DLS_Data'] \
            ['DLS0001']['ExperimentAveraged']['AverageCorrelation'] \
            ['Intensity']['Data'][:]
        inten[:, 0] *= self.factor
        df = pd.DataFrame(inten, columns = ['hydrodynamic_diameter',
                                            'amplitude'])
        df['uncle_dls_summary_id'] = self.well_id_to_summary(well)
        return df

    def dls_mass(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        pd.DataFrame
            Mass hydrodynamic diameter(nm) and amplitude for a single well
        """
        well_num = self.well_name_to_num(well)
        mass = self.file['Application1']['Run1'][well_num]['DLS_Data'] \
            ['DLS0001']['ExperimentAveraged']['AverageCorrelation'] \
            ['Mass']['Data'][:]
        mass[:, 0] *= self.factor
        df = pd.DataFrame(mass, columns = ['hydrodynamic_diameter',
                                           'amplitude'])
        return df

    # ----------------------------------------------------------------------- #
    # DATA COLLECTION FOR DLS SUMMARY                                         #
    # ----------------------------------------------------------------------- #
    @staticmethod
    def dls_summary_color():
        """
        NOTE: Datasets have not included this yet, therefore unable to locate
              where it is captured in .uni file.

        Returns
        -------
        np.nan
        """
        return np.nan

    def dls_summary_temperatures(self):
        """
        Returns
        -------
        pd.Series
            unit: °C
            Temperatures used in DLS analysis for all wells
        """
        wells = self.wells()
        temps = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            temps = np.append(temps,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data']['DLS0001']
                              ['ExperimentAveraged']['AverageCorrelation'].
                              attrs['Temperature'])
        return pd.Series(temps)

    def dls_summary_z_avg_diam(self, raw = True, diam = True):
        """
        NOTE: Modified values have upper limit of 1000, therefore any value
            greater than 1000 will be returned as 1000 if raw = False

        Parameters
        ----------
        raw : bool (default = True)
            Return raw or modified values
            See NOTE above for values greater than 1000
        diam : bool (default = True)
            Return diameters (True) or radii (False)

        Returns
        -------
        pd.Series
            unit: nanometer (nm)
        """
        wells = self.wells()
        vals = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            val = self.file['Application1']['Run1'][well_num] \
                ['DLS_Data']['DLS0001'] \
                ['ExperimentAveraged']['AverageCorrelation'] \
                .attrs['Radius']
            if diam:
                val = val * self.factor
            if not raw and val > 1000:
                val = 1000
            vals = np.append(vals, val)
        return pd.Series(vals)

    def dls_summary_z_avg_diff_coeff(self):
        """
        Uses Stokes-Einstein equation:

        D = (k * T) / (6 * π * η * r)

            k = Boltzmann's constant
            T = absolute temperature
            η = dynamic viscosity
            r = radius of particle

        Returns
        -------
        pd.Series
            Z-average differential coefficient for all wells
        """
        abs_temp = convert_temperature(self.dls_summary_temperatures(),
                                       'Celsius', 'Kelvin')  # C to K
        rad_m = self.dls_summary_z_avg_diam(diam = False) / 1000000000
        # nm to m
        visco = self.dls_summary_viscosity() / 1000  # cP to kg/m-s
        coef = (Boltzmann * abs_temp) / (6 * np.pi * visco * rad_m)
        return coef

    def dls_summary_stdev_diam(self, raw = True):
        """
        NOTE: Modified values have upper limit of 1000, therefore any value
            greater than 1000 will be returned as 1000 if raw = False

        Parameters
        ----------
        raw : bool (default = True)
            Return raw or modified values
            See NOTE above for values greater than 1000

        Returns
        -------
        pd.Series
            unit: nanometer (nm)
        """
        wells = self.wells()
        diams = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            radius = self.file['Application1']['Run1'][well_num] \
                ['DLS_Data']['DLS0001'] \
                ['ExperimentAveraged']['AverageCorrelation'] \
                .attrs['StdDev']
            diam = radius * self.factor
            if not raw and diam > 1000:
                diam = 1000
            diams = np.append(diams, diam)
        return pd.Series(diams)

    def dls_summary_pdi(self):
        """
        PDI = (σ/d)^2
        σ = standard deviation
        d = mean particle diameter

        Returns
        -------
        pd.Series
            PDI (polydispersity index) values for all wells
        """
        z_avg_diams = self.dls_summary_z_avg_diam(raw = True, diam = True)
        stdev_diams = self.dls_summary_stdev_diam(raw = True)
        pdis = []
        for s, z in zip(stdev_diams, z_avg_diams):
            pdis = np.append(pdis, ((s / z) ** 2))
        return pd.Series(pdis)

    def dls_summary_correlation_values(self, for_plotting = False):
        """
        Returns
        -------
        np.array
            Experimental (true) values,
            Expected (predicated) values,
            (Optional: time points)
        """
        wells = self.wells()
        values = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            corr_data = self.file['Application1']['Run1'][well_num] \
                ['DLS_Data']['DLS0001']['ExperimentAveraged'] \
                ['AverageCorrelation']['Correlations'][:]
            corr = corr_data[:, 0]
            time = corr_data[:, 1]
            min_of_corr = np.max(corr) / 100  # arbitrary cutoff: 1% of max
            # Assumption: amplitude is never >1% of max after it goes below
            true_values = corr[corr > min_of_corr]
            time_rel = time[:len(true_values)]

            popt, pcov = curve_fit(func, time_rel, true_values)
            predicated_values = func(time_rel, popt[0], popt[1])
            if for_plotting:
                values.append([true_values, predicated_values, time_rel])
            else:
                values.append([true_values, predicated_values])
        return np.array(values, dtype = object)

    def dls_summary_residuals(self):
        """
        Returns
        -------
        np.array
            Residuals for correlation fit for all wells
        """
        true_predicted_values = self.dls_summary_correlation_values()
        resid = []
        for i in true_predicted_values:
            diff = i[1] - i[0]
            resid.append(diff.tolist())
        return resid

    def dls_summary_rmse(self, mse = False):
        """
        NOTE: Exported files use a value called "Fit Var". It was instead
              decided to use RMSE as the measure of fit.

        Fit Var =
        (Residuals / (number of points – 4)) * amplitude factor * delay factor

        Parameters
        ----------
        mse : bool (Default = False)
            Root mean square error (False) or mean square error (True)

        Returns
        -------
        np.array
            Root mean squared errors for correlation fit for all wells
        """
        true_predicted_values = self.dls_summary_correlation_values()
        # true_values, predicted_values
        rmse = []
        for i in true_predicted_values:
            rmse = np.append(rmse,
                             mean_squared_error(i[0], i[1], squared = mse))
        return rmse

    def dls_summary_intensity(self):
        """
        Returns
        -------
        pd.Series
            Intensities (cps) for all wells
        """
        wells = self.wells()
        inten = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            inten = np.append(inten,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data']['DLS0001']
                              ['ExperimentAveraged']['AverageCorrelation'].
                              attrs['AverageIntensity'])
        return pd.Series(inten)

    def dls_summary_pk_mode_diam(self, raw = True, diam = True):
        """
        NOTE: Modified values have upper limit of 1000, therefore any value
            greater than 1000 will be returned as 1000 if raw = False

        Parameters
        ----------
        raw : bool (default = True)
            Return raw or modified values
            See NOTE above for values greater than 1000
        diam : bool (default = True)
            Return diameters (True) or radii (False)

        Returns
        -------
        pd.DataFrame
            Peak modal diameters for all peaks for all wells
        """
        wells = self.wells()
        diams = {}
        for well in wells:
            well_num = self.well_name_to_num(well)
            well_id = self.well_name_to_id(well)
            peaks = [i for i in self.file['Application1']['Run1'][well_num]
                     ['DLS_Data']['DLS0001']['ExperimentAveraged']
                     ['AverageCorrelation']['Intensity'].keys()
                     if 'peak' in i.lower()]
            for peak in peaks:
                val = self.file['Application1']['Run1'][well_num] \
                    ['DLS_Data']['DLS0001']['ExperimentAveraged'] \
                    ['AverageCorrelation']['Intensity'][peak] \
                    .attrs['Max'].item()
                diams.setdefault(well_id, []).append(val)
        df = pd.DataFrame.from_dict(diams, orient = 'index')
        df = df.rename(columns = {i: 'pk_{}_mode_diameter'.format(i + 1)
                                  for i in df.columns})
        if diam:
            df = 2 * df
        if not raw:
            df[df > 1000] = 1000
        return df

    def dls_summary_pk_est_mw(self, raw = True):
        """
        NOTE: What is the highest value before "Out of Range"?
              So far, the highest value seen that still displays: 23,983.70

        NOTE: Modified values have upper limit of ????, therefore any value
            greater than ???? will be returned as -1 if raw = False

        Parameters
        ----------
        raw : bool (default = True)
            Return raw or modified values
            See NOTE above for values greater than ????

        Returns
        -------
        pd.DataFrame
            Estimated molecular weights for all peaks for all wells
        """
        rad_df = self.dls_summary_pk_mode_diam(raw = True, diam = False)
        cols = {i: i.replace('mode_diameter', 'est_mw') if 'mode_diameter' in i
                else i for i in rad_df.columns}
        rad_df = rad_df.rename(columns = cols)
        mw_df = 2.75 * (rad_df**2.49)
        if not raw:
            mw_df[mw_df > 25000] = -1  # Adjust accordingly to note above
        return mw_df

    def dls_summary_pk_poly(self):
        """
        peak polydispersity (%) = (peak std / peak mean) * 100

        Returns
        -------
        pd.DataFrame
            Polydispersity percentage for all peaks for all wells
        """
        wells = self.wells()
        pk_poly = {}
        for well in wells:
            well_num = self.well_name_to_num(well)
            well_id = self.well_name_to_id(well)
            path = self.file['Application1']['Run1'][well_num] \
                ['DLS_Data']['DLS0001']['ExperimentAveraged'] \
                ['AverageCorrelation']['Intensity']
            peaks = [i for i in path.keys() if 'peak' in i.lower()]
            for peak in peaks:
                std = path[peak].attrs['Std'].item()
                mean = path[peak].attrs['Mean'].item()
                pk_poly.setdefault(well_id, []).append(100 * std / mean)
        df = pd.DataFrame.from_dict(pk_poly, orient = 'index')
        df = df.rename(columns = {i: 'pk_{}_polydispersity'.format(i + 1)
                                  for i in df.columns})
        return df

    def dls_summary_pk_mass(self):
        """
        Returns
        -------
        TODO: need to figure this out ASAP
        """
        return pd.Series(np.nan)

    def dls_summary_data_filter(self):
        """
        Returns
        -------
        pd.Series
            Filter used in DLS analysis for all wells
        """
        wells = self.wells()
        dataf = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            dataf = np.append(dataf,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data'].attrs['Data Filter Name'].
                              decode('utf-8'))
        return pd.Series(dataf)

    def dls_summary_viscosity(self):
        """
        Returns
        -------
        pd.Series
            unit: centipoise (cP)
            Viscosity found in DLS analysis for all wells
        """
        wells = self.wells()
        visco = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            visco = np.append(visco,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data']['DLS0001'].attrs['Viscosity'])
        return pd.Series(visco)

    def dls_summary_ri(self):
        """
        Returns
        -------
        pd.Series
            Refractive index used in DLS analysis for all wells
        """
        wells = self.wells()
        refin = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            refin = np.append(refin,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data']['DLS0001'].
                              attrs['Refractive Index'])
        return pd.Series(refin)

    def dls_atten_perc(self):
        """
        Returns
        -------
        pd.Series
            Attenuation percentage used for all wells.
            Used for calculating derived intensity
            Returned as float percentage, i.e. 75% = 75.0
        """
        wells = self.wells()
        atten = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            atten = np.append(atten,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data']['DLS0001'].
                              attrs['Attenuation %'])
        return pd.Series(atten)

    def dls_laser_perc(self):
        """
        Returns
        -------
        pd.Series
            Laser percentage used for all wells.
            Used for calculating derived intensity
            Returned as float percentage, i.e. 75% = 75.0
        """
        wells = self.wells()
        laser = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            laser = np.append(laser,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data']['DLS0001'].
                              attrs['Laser %'])
        return pd.Series(laser)

    def dls_summary_der_intensity(self):
        """
        Derived intensity = intensity / attenuation % / laser %

        Returns
        -------
        pd.Series
            unit: counts per second (cps)
            Derived intensities for all wells
        """
        inten = self.dls_summary_intensity()
        atten = self.dls_atten_perc() / 100
        laser = self.dls_laser_perc() / 100
        dis = []
        for i, a, l in zip(inten, atten, laser):
            dis = np.append(dis, (i / a / l))
        return pd.Series(dis)

    def dls_summary_min_pk_area(self):
        """
        Returns
        -------
        pd.Series
            Minimum peak area used in DLS analysis for all wells
            Returned as float percentage, i.e. 75% = 75.0
        """
        wells = self.wells()
        minpa = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            minpa = np.append(minpa,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data'].attrs['Minimum Area'])
        return pd.Series(minpa)

    def dls_summary_min_rh(self):
        """
        Returns
        -------
        pd.Series
            Minimum relative humidity used in DLS analysis for all wells
        """
        wells = self.wells()
        minrh = []
        for well in wells:
            well_num = self.well_name_to_num(well)
            minrh = np.append(minrh,
                              self.file['Application1']['Run1'][well_num]
                              ['DLS_Data'].attrs['Minimum Rh'])
        return pd.Series(minrh)

    # ----------------------------------------------------------------------- #
    # DATAFRAME ASSEMBLY                                                      #
    # ----------------------------------------------------------------------- #
    def dls_summary(self):
        """
        Returns
        -------
        pd.DataFrame
            Full dataframe of DLS summary data for all wells
        """
        well_ids = [self.well_name_to_id(well) for well in self.wells()]

        data = {
            'well_id'           : well_ids,
            'sample'            : self.samples(),
            'temperature'       : self.dls_summary_temperatures(),
            'z_avg_diameter'    : self.dls_summary_z_avg_diam(raw = False,
                                                              diam = True),
            'z_avg_diff_coeff'  : self.dls_summary_z_avg_diff_coeff(),
            'stdev_diameter'    : self.dls_summary_stdev_diam(raw = False),
            'pdi'               : self.dls_summary_pdi(),
            'intensity'         : self.dls_summary_intensity(),
            'data_filter'       : self.dls_summary_data_filter(),
            'viscosity'         : self.dls_summary_viscosity(),
            'refractive_index'  : self.dls_summary_ri(),
            'derived_intensity' : self.dls_summary_der_intensity(),
            'min_pk_area'       : self.dls_summary_min_pk_area(),
            'min_rel_humid'     : self.dls_summary_min_rh(),
            'color'             : self.dls_summary_color(),
            'residuals'         : self.dls_summary_residuals(),
            'rmse'              : self.dls_summary_rmse(),
        }

        # Multiple peaks, therefore below are returned as dataframes
        pk_mode_diam      = self.dls_summary_pk_mode_diam(raw = False,
                                                          diam = True)
        pk_est_mw         = self.dls_summary_pk_est_mw(raw = False)
        pk_polydispersity = self.dls_summary_pk_poly()
        pk_mass           = self.dls_summary_pk_mass()

        df = pd.DataFrame(data)
        df = df.merge(pk_mode_diam, right_index = True, left_on = 'well_id').\
            merge(pk_est_mw, right_index = True, left_on = 'well_id').\
            merge(pk_polydispersity, right_index = True, left_on = 'well_id')

        # TODO add mass

        return df

    # ----------------------------------------------------------------------- #
    # WRITE DATA TO POSTGRESQL                                                #
    # ----------------------------------------------------------------------- #
    def write_dls_summary_sql(self):
        """
        Returns
        -------
        None
        """
        self.exp_confirm_created()

        df = self.dls_summary()
        df.name = 'summary'
        df = self.df_to_sql(df)
        df.to_sql('uncle_dls_summary',
                  self.engine,
                  if_exists = 'append',
                  index = False)

    def write_dls_correlation_sql(self):
        """
        Returns
        -------
        None
        """
        self.exp_confirm_created()

        wells = self.wells()
        df = pd.DataFrame(columns =
                          ['uncle_dls_summary_id', 'time', 'amplitude'])
        for well in wells:
            corr_df = self.dls_correlation(well)
            df = df.append(corr_df).reset_index(drop = True)
        df.name = 'dls_correlation'
        df = self.df_to_sql(df)
        df.to_sql('uncle_dls_correlation',
                  self.engine,
                  if_exists = 'append',
                  index = False)

    def write_dls_intensity_sql(self):
        """
        Returns
        -------
        None
        """
        self.exp_confirm_created()

        wells = self.wells()
        df = pd.DataFrame(columns =
                          ['uncle_dls_summary_id', 'hydrodynamic_diameter',
                           'amplitude'])
        for well in wells:
            inten_df = self.dls_intensity(well)
            df = df.append(inten_df).reset_index(drop = True)
        df.name = 'dls_intensity'
        df = self.df_to_sql(df)
        df.to_sql('uncle_dls_intensity',
                  self.engine,
                  if_exists = 'append',
                  index = False)


def func(x, a, b):
    return a * np.exp(b * x)


def test_overlay(time_rel, true_values, residuals = None):
    popt, pcov = curve_fit(func, time_rel, true_values)
    calc_vals = func(time_rel, popt[0], popt[1])
    fig, ax1 = plt.subplots()
    if residuals is not None:
        ax1.plot(time_rel, true_values, time_rel, calc_vals)
        ax2 = ax1.twinx()
        ax2.plot(time_rel, residuals, 'r-.')
    else:
        ax1.plot(time_rel, true_values, time_rel, calc_vals)
    plt.show()


"""
h3 = DLS('/Users/jmiller/Desktop/UNcle Files/uni files/210602-01-Seq1 Cas9-pH003R.uni')
h4 = DLS('/Users/jmiller/Desktop/UNcle Files/uni files/Gen6 uni 1,2,3.uni')
save_directory = '/Users/jmiller/Desktop/UNcle Files/Misc/'
"""
