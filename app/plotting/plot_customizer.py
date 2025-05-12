from matplotlib.pyplot import Axes, rc_context

class PlotCustomizer:
    def __init__(
            self,
            axes: Axes,
            title: str = "",
            xlabel: str = "",
            ylabel: str = "",
            zlabel: str = "",
            xlim = None,
            ylim = None,
            zlim = None,
            grid: bool = False):
        
        self._custom_rc_params = {
            'text.usetex': True,
            'font.family': 'serif, sans-serif',
            'font.size': 11.,
            'mathtext.fontset': 'dejavusans', # https://matplotlib.org/stable/gallery/text_labels_and_annotations/mathtext_fontfamily_example.html
            'xtick.direction': 'in',
            'xtick.major.size': 5,
            'xtick.major.width': 0.5,
            'xtick.minor.size': 2.5,
            'xtick.minor.width': 0.5,
            'xtick.minor.visible': True,
            'xtick.top': True,
            'ytick.direction': 'in',
            'ytick.major.size': 5,
            'ytick.major.width': 0.5,
            'ytick.minor.size': 2.5,
            'ytick.minor.width': 0.5,
            'ytick.minor.visible': True,
            'ytick.right': True,
        }

        self.axes_object = axes
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.grid = grid

        self._apply_customizations()

    def _apply_customizations(self):

        with rc_context(rc = self._custom_rc_params):

            # (1): Set the Title -- if it's not there, will set empty string:
            self.axes_object.set_title(self.title)

            # (2): Set the X-Label -- if it's not there, will set empty string:
            self.axes_object.set_xlabel(self.xlabel)

            # (3): Set the Y-Label -- if it's not there, will set empty string:
            self.axes_object.set_ylabel(self.ylabel)

            # (4): Set the X-Limit, if it's provided:
            if self.xlim:
                self.axes_object.set_xlim(self.xlim)

            # (5): Set the Y-Limit, if it's provided:
            if self.ylim:
                self.axes_object.set_ylim(self.ylim)

            # (6): Check if the Axes object is a 3D Plot that has 'set_zlabel' method:
            if hasattr(self.axes_object, 'set_zlabel'):

                # (6.1): If so, set the Z-Label -- if it's not there, will set empty string:
                self.axes_object.set_zlabel(self.zlabel)

            # (7): Check if the Axes object is 3D again and has a 'set_zlim' method:
            if self.zlim and hasattr(self.axes_object, 'set_zlim'):

                # (7.1): If so, set the Z-Limit, if it's provided:
                self.axes_object.set_zlim(self.zlim)

            # (8): Apply a grid on the plot according to a boolean flag:
            self.axes_object.grid(self.grid)

    def add_line_plot(self, x_data, y_data, label: str = "", color = None, linestyle = '-'):
        """
        Add a line plot to the Axes object:
        connects element-wise points of the two provided arrays.

        Parameters
        ----------
        x_data: array_like
            
        y_data: array_like

        label: str

        color: str

        linestyle: str
        """

        with rc_context(rc = self._custom_rc_params):

            # (1): Just add the line plot:
            self.axes_object.plot(x_data, y_data, label = label, color = color, linestyle = linestyle)

            if label:
                self.axes_object.legend()

    def add_scatter_plot(self, x_data, y_data, radial_size: float = 1., label: str = "", color = None, marker = 'o'):
        """
        Add a scatter plot to the Axes object.

        Parameters
        ----------
        x_data: array_like
            
        y_data: array_like

        radial_size: float
        
        label: str

        color: str 

        marker: str
        """

        with rc_context(rc = self._custom_rc_params):

            # (1): Add the scatter plot:
            self.axes_object.scatter(
                x_data,
                y_data,
                s = radial_size,
                label = label,
                color = color,
                marker = marker)

            if label:
                self.axes_object.legend()

    def add_errorbar_plot(
            self,
            x_data,
            y_data,
            y_errorbars,
            x_errorbars,
            label: str = "",
            color = 'black',
            marker = 'o'):
        """
        Add a scatter plot with errorbars to the Axes object.

        Parameters
        ----------
        x_data: array_like
            
        y_data: array_like

        x_errorbars: array_like
            
        y_errorbars: array_like

        label: str

        color: str |

        marker: str
        """

        with rc_context(rc = self._custom_rc_params):

            # (1): Add the errorbar plot:
            self.axes_object.errorbar(
                x = x_data,
                y = y_data, 
                yerr = y_errorbars,
                xerr = x_errorbars,
                label = label,
                color = color,
                marker = marker,
                linestyle = '', 
                markersize = 1.0,
                ecolor = 'black',
                elinewidth = 0.5,
                capsize = 1)

            if label:
                self.axes_object.legend()

    def add_bar_plot(self, x_positions, y_data_heights, label = "", color = None):

        with rc_context(rc = self._custom_rc_params):

            # (1): Add the bar plot:
            self.axes_object.bar(x_positions, y_data_heights, label = label, color = color)

            if label:
                self.axes_object.legend()

    def add_3d_scatter_plot(self, x_data, y_data, z_data, color = None, marker = 'o'):

        with rc_context(rc = self._custom_rc_params):

            # (1): Plot points in R3:
            self.axes_object.scatter(x_data, y_data, z_data, color = color, marker = marker)

    def add_surface_plot(self, X, Y, Z, colormap: str ='viridis'):

        with rc_context(rc = self._custom_rc_params):

            # (1): Plot as surface in R3:
            self.axes_object.plot_surface(X, Y, Z, cmap = colormap, antialiased=False)