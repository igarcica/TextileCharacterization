
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # import numpy as np

# # # Sample data structure: Replace this with your actual data
# # data = {
# #     'Attribute': ['Friction', 'Stiffness', 'Elasticity', 'Material', 'Shape',
# #                   'Weight', 'Size', 'Color', 'Construction'],
# #     'EOS': [63, 70, 140, 80, 70, 1, 50, 25, 4],
# #     'HCOS': [40, 90, 20, 30, 40, 70, 100, 10, 4],
# #     'DOS': [20, 80, 130, 50, 60, 40, 10, 5, 4]
# # }

# # # 	Elastic Object Set (EOS)	Household Cloth Object Set (HCOS)	Dressing Object Set (DOS)	Scale
# # # Friction	63	31	14	100
# # # Stiffness	53	71	67	100
# # # Elasticity	106	22	103	150
# # # Material	7	2	4	10
# # # Shape	1	3	6	9
# # # Weight	24	704	442	1000
# # # Size	31	250	87	400
# # # Color	23	5	4	35
# # # Construction	2	2	2	3

# # df = pd.DataFrame(data)
# # categories = df['Attribute'].tolist()
# # labels = df.columns[1:]

# # # Normalize angles for radar chart
# # N = len(categories)
# # angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
# # angles += angles[:1]  # repeat the first angle to close the circle

# # # Start figure
# # fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# # # Plot each dataset
# # colors = ['green', 'royalblue', 'darkorange']
# # for idx, label in enumerate(labels):
# #     values = df[label].tolist()
# #     values += values[:1]  # repeat the first value to close the shape
# #     ax.plot(angles, values, label=label, color=colors[idx])
# #     ax.fill(angles, values, alpha=0.25, color=colors[idx])

# # # Set category labels
# # ax.set_xticks(angles[:-1])
# # ax.set_xticklabels(categories)

# # # Optional: adjust r-labels and layout
# # ax.set_rlabel_position(30)
# # ax.tick_params(colors='gray')
# # plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3)

# # plt.tight_layout()
# # plt.show()



# import matplotlib.pyplot as plt
# import numpy as np

# from matplotlib.patches import Circle, RegularPolygon
# from matplotlib.path import Path
# from matplotlib.projections import register_projection
# from matplotlib.projections.polar import PolarAxes
# from matplotlib.spines import Spine
# from matplotlib.transforms import Affine2D


# def radar_factory(num_vars, frame='polygon'):
#     theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

#     class RadarTransform(PolarAxes.PolarTransform):
#         def transform_path_non_affine(self, path):
#             if path._interpolation_steps > 1:
#                 path = path.interpolated(num_vars)
#             return Path(self.transform(path.vertices), path.codes)

#     class RadarAxes(PolarAxes):
#         name = 'radar'
#         PolarTransform = RadarTransform

#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#             self.set_theta_zero_location('N')

#         def fill(self, *args, closed=True, **kwargs):
#             return super().fill(closed=closed, *args, **kwargs)

#         def plot(self, *args, **kwargs):
#             lines = super().plot(*args, **kwargs)
#             for line in lines:
#                 self._close_line(line)

#         def _close_line(self, line):
#             x, y = line.get_data()
#             if x[0] != x[-1]:
#                 x = np.append(x, x[0])
#                 y = np.append(y, y[0])
#                 line.set_data(x, y)

#         def set_varlabels(self, labels):
#             self.set_thetagrids(np.degrees(theta), labels)

#         def _gen_axes_patch(self):
#             if frame == 'circle':
#                 return Circle((0.5, 0.5), 0.5)
#             elif frame == 'polygon':
#                 return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
#             else:
#                 raise ValueError(f"Unknown value for 'frame': {frame}")

#         def _gen_axes_spines(self):
#             if frame == 'circle':
#                 return super()._gen_axes_spines()
#             elif frame == 'polygon':
#                 spine = Spine(axes=self, spine_type='circle',
#                               path=Path.unit_regular_polygon(num_vars))
#                 spine.set_transform(
#                     Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
#                 return {'polar': spine}
#             else:
#                 raise ValueError(f"Unknown value for 'frame': {frame}")

#     register_projection(RadarAxes)
#     return theta


# def get_custom_data():
#     labels = ['Friction', 'Stiffness', 'Elasticity', 'Material', 'Shape', 
#               'Weight', 'Size', 'Color', 'Construction']
    
#     EOS = [60, 70, 140, 80, 70, 1, 50, 25, 4]
#     HCOS = [40, 90, 20, 30, 40, 70, 100, 10, 4]
#     DOS = [20, 80, 130, 50, 60, 40, 10, 5, 4]
    
#     # Per-axis max values for normalization (can be customized)
#     axis_ranges = [100, 100, 150, 100, 100, 100, 400, 30, 5]
    
#     return labels, [EOS, HCOS, DOS], ['Elastic Object Set (EOS)', 'Household Cloth Object Set (HCOS)', 'Dressing Object Set (DOS)'], axis_ranges


# if __name__ == '__main__':
#     labels, data_sets, legends, axis_max = get_custom_data()
#     N = len(labels)
#     theta = radar_factory(N, frame='polygon')

#     # Normalize data
#     normalized_data = []
#     for data in data_sets:
#         normalized_data.append([val / max_val for val, max_val in zip(data, axis_max)])

#     # Plot
#     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
#     fig.subplots_adjust(top=0.85, bottom=0.15)

#     colors = ['mediumseagreen', 'royalblue', 'darkorange']
#     for data, label, color in zip(normalized_data, legends, colors):
#         ax.plot(theta, data, color=color, linewidth=2, label=label)
#         ax.fill(theta, data, color=color, alpha=0.25)

#     ax.set_varlabels(labels)
#     ax.set_title("Radar Chart with Per-Axis Units", weight='bold', size=14, position=(0.5, 1.1))
#     ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize='small')

#     # Add radial labels manually per axis
#     for i, (label, max_val) in enumerate(zip(labels, axis_max)):
#         angle_rad = theta[i]
#         ax.text(angle_rad, 1.05, f'{max_val}', size=8, ha='center', va='center')

#     plt.show()





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


# ===== Custom RadarAxes Class (standalone) ===== #
class RadarAxes(PolarAxes):
    name = 'radar'

    def __init__(self, *args, num_vars=3, frame='polygon', **kwargs):
        self.num_vars = num_vars
        self.frame = frame
        self.theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        super().__init__(*args, **kwargs)
        self.set_theta_zero_location('N')

    def fill(self, *args, closed=True, **kwargs):
        return super().fill(closed=closed, *args, **kwargs)

    def plot(self, *args, **kwargs):
        lines = super().plot(*args, **kwargs)
        for line in lines:
            self._close_line(line)
        return lines

    def _close_line(self, line):
        x, y = line.get_data()
        if x[0] != x[-1]:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            line.set_data(x, y)

    def set_varlabels(self, labels):
        self.set_thetagrids(np.degrees(self.theta), labels)

    def _gen_axes_patch(self):
        if self.frame == 'circle':
            return Circle((0.5, 0.5), 0.5)
        elif self.frame == 'polygon':
            return RegularPolygon((0.5, 0.5), self.num_vars, radius=.5, edgecolor="k")
        raise ValueError(f"Unknown frame: {self.frame}")

    def _gen_axes_spines(self): #Plot polygon contour
        if self.frame == 'polygon':
            spine = Spine(axes=self, spine_type='circle', path=Path.unit_regular_polygon(self.num_vars))
            spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
            return {'polar': spine}
        return super()._gen_axes_spines()


register_projection(RadarAxes)

# # ===== Sample Data with Per-Axis Max Values ===== #
# # properties = ['Friction', 'Stiffness', 'Elasticity', 'Material', 'Shape', 'Weight', 'Size', 'Color', 'Construction']
# properties = ['Friction', 'Construction', 'Color', 'Size', 'Weight','Shape','Material','Elasticity','Stiffness',]

# EOS = [60, 70, 140, 80, 70, 1, 50, 25, 4]
# HCOS = [40, 90, 20, 30, 40, 70, 100, 10, 4]
# DOS = [20, 80, 130, 50, 60, 40, 10, 5, 4]

# object_sets = [EOS, HCOS, DOS]
# object_set_names = ['Elastic Object Set (EOS)', 'Household Cloth Object Set (HCOS)', 'Dressing Object Set (DOS)']
# axis_max = [100, 100, 150, 100, 100, 100, 400, 30, 5]

# #  # Sample data structure: Replace this with your actual data
# # data = {
# #     'Properties': ['Friction', 'Stiffness', 'Elasticity', 'Material', 'Shape', 'Weight', 'Size', 'Color', 'Construction'],
# #     'EOS': [63, 70, 140, 80, 70, 1, 50, 25, 4],
# #     'HCOS': [40, 90, 20, 30, 40, 70, 100, 10, 4],
# #     'DOS': [20, 80, 130, 50, 60, 40, 10, 5, 4]
# # }
# # df = pd.DataFrame(data)
# # categories = df['Properties'].tolist()
# # labels = df.columns[1:]

df = pd.read_csv("object_sets_data.csv", index_col=0)

# axis_max = df.max(axis=1).tolist()
axis_max = df["Range"].tolist() # Extract 'Max' column values for normalization
print(axis_max)
df = df.drop(columns="Range") # Drop the 'Max' column to keep only dataset columns

properties = df.index.tolist()
print(properties)

object_set_names = df.columns.tolist()
print(object_set_names)

object_sets = [df[col].tolist() for col in df.columns] # list of lists, one list per dataset
print(object_sets)

# Normalize
normalized = [[val / max_val for val, max_val in zip(dataset, axis_max)]
              for dataset in object_sets]

# Pass labels, normalized, datasets, legend_labels, axis_max into your radar plotting code

# # Normalize data
# normalized = []
# for data in object_sets:
#     normalized.append([val / max_val for val, max_val in zip(data, axis_max)])


# ===== Plot Radar Chart ===== #
fig, ax = plt.subplots(figsize=(8, 8),
                       subplot_kw=dict(projection='radar', num_vars=len(properties), frame='polygon'))

colors = ['mediumseagreen', 'royalblue', 'darkorange']
theta = ax.theta

# for values, label, color in zip(normalized, object_set_names, colors):
#     ax.plot(theta, values, color=color, linewidth=2, label=label)
#     ax.fill(theta, values, color=color, alpha=0.25)

for points, values, label, color in zip(object_sets, normalized, object_set_names, colors):
    ax.plot(theta, values, color=color, linewidth=2, label=label)
    ax.fill(theta, values, color=color, alpha=0.25)

    print(points)
    # Highlight the maximum point
    max_idx = np.argmax(points)
    print(max_idx)
    ax.plot(theta, values, 'o', color=color, markersize=8)


ax.set_varlabels(properties)
ax.set_title("Radar Chart with Per-Axis Units", weight='bold', size=14, position=(0.5, 1.1))
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize='small')

# Add per-axis max value annotations
for angle, max_val in zip(theta, axis_max):
    ax.text(angle, 1.05, str(max_val), ha='center', va='center', fontsize=8)

plt.show()





# import matplotlib.pyplot as plt
# import numpy as np

# from matplotlib.patches import Circle, RegularPolygon
# from matplotlib.path import Path
# from matplotlib.projections import register_projection
# from matplotlib.projections.polar import PolarAxes
# from matplotlib.spines import Spine
# from matplotlib.transforms import Affine2D


# def radar_factory(num_vars, frame='circle'):
#     """
#     Create a radar chart with `num_vars` Axes.

#     This function creates a RadarAxes projection and registers it.

#     Parameters
#     ----------
#     num_vars : int
#         Number of variables for radar chart.
#     frame : {'circle', 'polygon'}
#         Shape of frame surrounding Axes.

#     """
#     # calculate evenly-spaced axis angles
#     theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

#     class RadarTransform(PolarAxes.PolarTransform):

#         def transform_path_non_affine(self, path):
#             # Paths with non-unit interpolation steps correspond to gridlines,
#             # in which case we force interpolation (to defeat PolarTransform's
#             # autoconversion to circular arcs).
#             if path._interpolation_steps > 1:
#                 path = path.interpolated(num_vars)
#             return Path(self.transform(path.vertices), path.codes)

#     class RadarAxes(PolarAxes):

#         name = 'radar'
#         PolarTransform = RadarTransform

#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#             # rotate plot such that the first axis is at the top
#             self.set_theta_zero_location('N')

#         def fill(self, *args, closed=True, **kwargs):
#             """Override fill so that line is closed by default"""
#             return super().fill(closed=closed, *args, **kwargs)

#         def plot(self, *args, **kwargs):
#             """Override plot so that line is closed by default"""
#             lines = super().plot(*args, **kwargs)
#             for line in lines:
#                 self._close_line(line)

#         def _close_line(self, line):
#             x, y = line.get_data()
#             # FIXME: markers at x[0], y[0] get doubled-up
#             if x[0] != x[-1]:
#                 x = np.append(x, x[0])
#                 y = np.append(y, y[0])
#                 line.set_data(x, y)

#         def set_varlabels(self, labels):
#             self.set_thetagrids(np.degrees(theta), labels)

#         def _gen_axes_patch(self):
#             # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
#             # in axes coordinates.
#             if frame == 'circle':
#                 return Circle((0.5, 0.5), 0.5)
#             elif frame == 'polygon':
#                 return RegularPolygon((0.5, 0.5), num_vars,
#                                       radius=.5, edgecolor="k")
#             else:
#                 raise ValueError("Unknown value for 'frame': %s" % frame)

#         def _gen_axes_spines(self):
#             if frame == 'circle':
#                 return super()._gen_axes_spines()
#             elif frame == 'polygon':
#                 # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
#                 spine = Spine(axes=self,
#                               spine_type='circle',
#                               path=Path.unit_regular_polygon(num_vars))
#                 # unit_regular_polygon gives a polygon of radius 1 centered at
#                 # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
#                 # 0.5) in axes coordinates.
#                 spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
#                                     + self.transAxes)
#                 return {'polar': spine}
#             else:
#                 raise ValueError("Unknown value for 'frame': %s" % frame)

#     register_projection(RadarAxes)
#     return theta


# def example_data():
#     # The following data is from the Denver Aerosol Sources and Health study.
#     # See doi:10.1016/j.atmosenv.2008.12.017
#     #
#     # The data are pollution source profile estimates for five modeled
#     # pollution sources (e.g., cars, wood-burning, etc) that emit 7-9 chemical
#     # species. The radar charts are experimented with here to see if we can
#     # nicely visualize how the modeled source profiles change across four
#     # scenarios:
#     #  1) No gas-phase species present, just seven particulate counts on
#     #     Sulfate
#     #     Nitrate
#     #     Elemental Carbon (EC)
#     #     Organic Carbon fraction 1 (OC)
#     #     Organic Carbon fraction 2 (OC2)
#     #     Organic Carbon fraction 3 (OC3)
#     #     Pyrolyzed Organic Carbon (OP)
#     #  2)Inclusion of gas-phase specie carbon monoxide (CO)
#     #  3)Inclusion of gas-phase specie ozone (O3).
#     #  4)Inclusion of both gas-phase species is present...
#     data = [
#         ['Sulfate', 'Nitrate', 'EC', 'OC1', 'OC2', 'OC3', 'OP', 'CO', 'O3'],
#         ('Basecase', [
#             [0.88, 0.01, 0.03, 0.03, 0.00, 0.06, 0.01, 0.00, 0.00],
#             [0.07, 0.95, 0.04, 0.05, 0.00, 0.02, 0.01, 0.00, 0.00],
#             [0.01, 0.02, 0.85, 0.19, 0.05, 0.10, 0.00, 0.00, 0.00],
#             [0.02, 0.01, 0.07, 0.01, 0.21, 0.12, 0.98, 0.00, 0.00],
#             [0.01, 0.01, 0.02, 0.71, 0.74, 0.70, 0.00, 0.00, 0.00]]),
#         ('With CO', [
#             [0.88, 0.02, 0.02, 0.02, 0.00, 0.05, 0.00, 0.05, 0.00],
#             [0.08, 0.94, 0.04, 0.02, 0.00, 0.01, 0.12, 0.04, 0.00],
#             [0.01, 0.01, 0.79, 0.10, 0.00, 0.05, 0.00, 0.31, 0.00],
#             [0.00, 0.02, 0.03, 0.38, 0.31, 0.31, 0.00, 0.59, 0.00],
#             [0.02, 0.02, 0.11, 0.47, 0.69, 0.58, 0.88, 0.00, 0.00]]),
#         ('With O3', [
#             [0.89, 0.01, 0.07, 0.00, 0.00, 0.05, 0.00, 0.00, 0.03],
#             [0.07, 0.95, 0.05, 0.04, 0.00, 0.02, 0.12, 0.00, 0.00],
#             [0.01, 0.02, 0.86, 0.27, 0.16, 0.19, 0.00, 0.00, 0.00],
#             [0.01, 0.03, 0.00, 0.32, 0.29, 0.27, 0.00, 0.00, 0.95],
#             [0.02, 0.00, 0.03, 0.37, 0.56, 0.47, 0.87, 0.00, 0.00]]),
#         ('CO & O3', [
#             [0.87, 0.01, 0.08, 0.00, 0.00, 0.04, 0.00, 0.00, 0.01],
#             [0.09, 0.95, 0.02, 0.03, 0.00, 0.01, 0.13, 0.06, 0.00],
#             [0.01, 0.02, 0.71, 0.24, 0.13, 0.16, 0.00, 0.50, 0.00],
#             [0.01, 0.03, 0.00, 0.28, 0.24, 0.23, 0.00, 0.44, 0.88],
#             [0.02, 0.00, 0.18, 0.45, 0.64, 0.55, 0.86, 0.00, 0.16]])
#     ]
#     return data


# if __name__ == '__main__':
#     N = 9
#     theta = radar_factory(N, frame='polygon')

#     data = example_data()
#     spoke_labels = data.pop(0)

#     fig, axs = plt.subplots(figsize=(9, 9), nrows=2, ncols=2,
#                             subplot_kw=dict(projection='radar'))
#     fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

#     colors = ['b', 'r', 'g', 'm', 'y']
#     # Plot the four cases from the example data on separate Axes
#     for ax, (title, case_data) in zip(axs.flat, data):
#         ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
#         ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
#                      horizontalalignment='center', verticalalignment='center')
#         for d, color in zip(case_data, colors):
#             ax.plot(theta, d, color=color)
#             ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
#         ax.set_varlabels(spoke_labels)

#     # add legend relative to top-left plot
#     labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
#     legend = axs[0, 0].legend(labels, loc=(0.9, .95),
#                               labelspacing=0.1, fontsize='small')

#     fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
#              horizontalalignment='center', color='black', weight='bold',
#              size='large')

#     plt.show()






import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='polygon'):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
            else:
                raise ValueError(f"Unknown value for 'frame': {frame}")

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self, spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(
                    Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError(f"Unknown value for 'frame': {frame}")

    register_projection(RadarAxes)
    return theta


def get_custom_data():
    labels = ['Friction', 'Stiffness', 'Elasticity', 'Material', 'Shape', 
              'Weight', 'Size', 'Color', 'Construction']
    
    EOS = [60, 70, 140, 80, 70, 1, 50, 25, 4]
    HCOS = [40, 90, 20, 30, 40, 70, 100, 10, 4]
    DOS = [20, 80, 130, 50, 60, 40, 10, 5, 4]
    
    # Per-axis max values for normalization (can be customized)
    axis_ranges = [100, 100, 150, 100, 100, 100, 400, 30, 5]
    
    return labels, [EOS, HCOS, DOS], ['Elastic Object Set (EOS)', 'Household Cloth Object Set (HCOS)', 'Dressing Object Set (DOS)'], axis_ranges


if __name__ == '__main__':
    labels, data_sets, legends, axis_max = get_custom_data()
    N = len(labels)
    theta = radar_factory(N, frame='polygon')

    # Normalize data
    normalized_data = []
    for data in data_sets:
        normalized_data.append([val / max_val for val, max_val in zip(data, axis_max)])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.15)

    colors = ['mediumseagreen', 'royalblue', 'darkorange']
    for data, label, color in zip(normalized_data, legends, colors):
        ax.plot(theta, data, color=color, linewidth=2, label=label)
        ax.fill(theta, data, color=color, alpha=0.25)

    ax.set_varlabels(labels)
    ax.set_title("Radar Chart with Per-Axis Units", weight='bold', size=14, position=(0.5, 1.1))
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize='small')

    # Add radial labels manually per axis
    for i, (label, max_val) in enumerate(zip(labels, axis_max)):
        angle_rad = theta[i]
        ax.text(angle_rad, 1.05, f'{max_val}', size=8, ha='center', va='center')

    plt.show()







