# Datasets

We provide a short descriptions of the example datasets we provide through the `f3rm-download-data` command.

___

**Table of Contents**

- [`panda`](#panda)
    - [`scene_001`](#scene_001)
    - [`scene_002`](#scene_002)
    - [`scene_003`](#scene_003)
- [`rooms`](#rooms)
    - [`george_the_panda`](#george_the_panda)

___

## `panda`

The `panda` datasets are taken by the robot using an Intel RealSense D415 mounted on a selfie stick. Each dataset
consists of 50 1280x720 RGB images of a scene with various objects collected around the lab in various poses.

### `scene_001`

<img src="images/dataset_previews/panda/scene_001.jpg" width="350" alt="panda/scene_001">

The teaser scene used throughout the paper, website and video.

**Objects Present:** transparent jug, mango, metal jug, Baymax plush toy, apple, blue screwdriver, red screwdriver,
plastic bowl, can of SPAM, grapes, whiteboard marker, wood blocks

### `scene_002`

<img src="images/dataset_previews/panda/scene_002.jpg" width="350" alt="panda/scene_002">

A scene we used for the language-guided manipulation results in the carousel on the website.

**Objects Present:** blue mug, measuring cup, pink mug, teddy bear, transparent jug, scissors, screwdriver, roll of
tape, blocks

### `scene_003`

<img src="images/dataset_previews/panda/scene_003.jpg" width="350" alt="panda/scene_003">

Another test scene we used for language-guided manipulation.

**Objects Present:** spatula, mug, water jug, blue screwdriver, black screwdriver, measuring beaker, roll of tape,
wood block

___

## `rooms`

The `rooms` datasets consist of room-scale scenes which we captured using our phones. These datasets similarly contain
interesting objects which you can query via language using CLIP feature fields.

### `george_the_panda`

<img src="images/dataset_previews/rooms/george_the_panda_1.jpg" width="400" alt="rooms/george_the_panda">&nbsp;
<img src="images/dataset_previews/rooms/george_the_panda_2.jpg" width="400" alt="rooms/george_the_panda">

This scene is of the room that George our Panda robot used to live in. The table and room contains a variety of objects,
including those from the YCB dataset. Some of these objects may not be entirely picked up by the CLIP feature field as
the source images do not get a good view of the object.

**Objects Present:**

- **On the table:** electric screwdriver, Cheez-It box, Domino Sugar box, mustard bottle, strawberries, lemon, soup
  cans, can of SPAM, can of tuna, brick, Lego toy, white bowl, tennis ball, baseball, WD-40 can, Jello box, Rubik's
  cube, cleaning spray, toy gun
- **In the room:** cabinet, office chairs, monitors
