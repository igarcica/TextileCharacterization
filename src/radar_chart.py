### Python script to generate the radar chart for benchmarking cloth object sets through its object properties variation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


## Radar chart with custom axes
class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(9)
            return Path(self.transform(path.vertices), path.codes)

class RadarAxes(PolarAxes):
    name = 'radar'
    PolarTransform = RadarTransform

    def __init__(self, *args, num_vars=9, frame='polygon', **kwargs):
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
        self.set_thetagrids(np.degrees(self.theta), labels,  fontsize=14)

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


df = pd.read_csv("object_sets_data.csv", index_col=0)

# axis_max = df.max(axis=1).tolist()
axis_max = df["Range"].tolist() # Extract 'Max' column values for normalization
print("Axis max range: ", axis_max)
df = df.drop(columns="Range") # Drop the 'Max' column to keep only dataset columns

properties = df.index.tolist()
print("Cloth properties: ", properties) 

object_set_names = df.columns.tolist()
print("Object set names: ", object_set_names)

object_sets = [df[col].tolist() for col in df.columns] 
# print(object_sets) #Properties values

normalized = [[val / max_val for val, max_val in zip(dataset, axis_max)]
              for dataset in object_sets]


# Plot radar chart
fig, ax = plt.subplots(figsize=(8, 8),
                       subplot_kw=dict(projection='radar', num_vars=len(properties), frame='polygon'))

colors = ['mediumseagreen', 'royalblue', 'darkorange']
theta = ax.theta

for points, values, label, color in zip(object_sets, normalized, object_set_names, colors):
    ax.plot(theta, values, color=color, linewidth=2, label=label)
    ax.fill(theta, values, color=color, alpha=0.25)

    # print(points)
    # max_idx = np.argmax(points) # Highlight the maximum point
    # print(max_idx)
    ax.plot(theta, values, 'o', color=color, markersize=8)



# ax.set_title("Cloth Object Sets comparison", weight='bold', size=14, position=(0.5, 1.1))
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize='small')
ax.set_yticklabels([])
ax.set_xticklabels([])
# ax.tick_params(axis='y', labelsize=20, pad=10)  #Axes
ax.set_varlabels(properties) #Axes names (properties)


# Axis range annotations
# for angle, max_val in zip(theta, axis_max):
#     ax.text(angle, 0.75, str(max_val), ha='center', va='center', fontsize=12)

plt.show()



