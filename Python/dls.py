from hdf5 import HDF5
import pandas as pd


class DLS(HDF5):
    """
    Subclass of HDF5 class, specific for DLS portion of experiment

    Attributes
    ----------
    Reference HDF5 class

    Methods
    -------

    """
    def __init__(self, file_path):
        super().__init__(file_path)
        # Hydrodynamic diameter is consistently 2x values found in .uni file
        # Is this possibly because values are radii and Excel is diameter?
        self.factor = 2

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
        df = pd.DataFrame(inten, columns = ['Hydrodynamic Diameter (nm)',
                                            'Amplitude'])
        return df

    def dls_mass(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            Mass hydrodynamic diameter(nm) and amplitude for a single well
        """
        well_num = self.well_name_to_num(well)
        mass = self.file['Application1']['Run1'][well_num]['DLS_Data'] \
            ['DLS0001']['ExperimentAveraged']['AverageCorrelation'] \
            ['Mass']['Data'][:]
        mass[:, 0] *= self.factor
        df = pd.DataFrame(mass, columns = ['Hydrodynamic Diameter (nm)',
                                            'Amplitude'])
        return df

    def dls_correlation(self, well):
        """
        Parameters
        ----------
        well : str
            Single well name, e.g. 'A1'

        Returns
        -------
        np.array
            Correlation time(s) and amplitude for a single well
        """
        well_num = self.well_name_to_num(well)
        corr = self.file['Application1']['Run1'][well_num]['DLS_Data'] \
            ['DLS0001']['ExperimentAveraged']['AverageCorrelation'] \
            ['Correlations'][:]
        # Swap columns to align with typical export
        corr = corr[:, [1, 0]]
        df = pd.DataFrame(corr, columns = ['Time (s)', 'Amplitude'])
        return df

    # ----------------------------------------------------------------------- #
    # WRITE DATA TO EXCEL                                                     #
    # ----------------------------------------------------------------------- #
    def write_dls_bundle_csv(self, save_directory):
        """
        Parameters
        ----------
        save_directory : str
            Directory to save CSVs to
            Each well has 3 files: Intensity, Mass, Correlation

        Returns
        -------
        None
        """
        wells = self.wells()
        for well in wells:
            inten_df = self.dls_intensity(well)
            mass_df = self.dls_mass(well)
            corr_df = self.dls_correlation(well)

            inten_df.to_csv(save_directory + '/Intensity-{}-15.csv'.
                            format(well), index = False)
            mass_df.to_csv(save_directory + '/Mass-{}-15.csv'.
                           format(well), index = False)
            corr_df.to_csv(save_directory + '/Correlation-{}-15.csv'.
                           format(well), index = False)


h3 = DLS('/Users/jmiller/Desktop/UNcle Files/uni files/210602-01-Seq1 Cas9-pH003R.uni')
h4 = DLS('/Users/jmiller/Desktop/UNcle Files/uni files/Gen6 uni 1,2,3.uni')
save_directory = '/Users/jmiller/Desktop/UNcle Files/Misc/'
