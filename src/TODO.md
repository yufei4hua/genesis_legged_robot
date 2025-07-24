# Train
## without imitation
```bash
git checkout 5441b0b7
python src/ainex_train.py -e no_imitation 
```
## with imitation
```bash
git checkout 459780bd
python src/ainex_train.py -e with_imitation 
```

### imitation 
with / without

### decrease lin. speed command
lin_vel_x_range

### collision 
```
<collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="package://ainex_description/meshes/l_ank_roll_link.STL"/>
      </geometry>
```
src/ainex_description/urdf/ainex.urdf, change from meshes to rectangle

### simulation frequency
