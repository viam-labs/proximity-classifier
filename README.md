# proximity-classifier

Viam vision service that reads a camera point cloud and returns one classification
for every configured point that has cloud points nearby. Several classes can match
the same frame.

Model: `viam-labs:vision:proximity-classifier`

`CaptureAllFromCamera` returns a grid image of the cloud, including when `classes`
is empty. That picture is how you choose coordinates. The default side view (`yz`)
draws +Y to the left and +Z up. Color is X, the axis that view collapses.

## Configuration

```json
{
  "camera": "cup-camera",
  "radius": 0.5,
  "classes": {
    "left-cup": { "x": 9.1, "y": 3, "z": -3 },
    "right-cup": { "x": 6.3, "y": -3, "z": -1.5 }
  }
}
```

Those two points are the short cup and the tall cup in the sample cloud
`testdata/cups.pcd`. `left-cup` is the short one. A radius of 0.5 reaches that
cup and misses the other. Coordinates are in the units of the camera's point
cloud. `GetPointCloud` PCD data is in meters.

### Attributes

| Name | Type | Inclusion | Description |
| --- | --- | --- | --- |
| `camera` | string | Required | Name of a camera that returns a point cloud. |
| `classes` | object | Optional | Map of label to `{x, y, z}`. Leave this empty while you are reading the grid. |
| `radius` | float | Optional | How near a cloud point must be to count. Defaults to 0.5. |
| `min_points` | int | Optional | How many points must fall inside the radius. Defaults to 1. |
| `view` | string | Optional | Grid projection: `yz` (default), `xz`, or `xy`. Pass `{"view": "xz"}` in the capture `extra` to switch for one call. |

## Reading the grid

Call `CaptureAllFromCamera` with `return_image` set. The image has tick marks on
the two drawn axes and a color bar for the collapsed axis. Once classes are
configured, each one is drawn as a circle of `radius`.
