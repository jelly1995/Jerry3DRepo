# Jerry3DRepo


How to add a class:
1. preprocess_nuscenes.py
modify get_mapping method

2. pointpillars_nuscenes.yml
modify configs
a. add classes
b. head.ranges and head.sizes

3. nuscenes.py
a. modify self.num_classes
b. modify get_label_to_names method