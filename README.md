**Sedimentary Rock Classification using CNN**
Sedimentary rocks are some of the most widely occurring rock types on Earth. They are formed through the deposition, compression, and cementation of sediments over millions of years. These rocks contain crucial information about Earth’s history, past environments, and natural resources.
Manual identification of sedimentary rocks is typically done by trained geologists who examine:

Texture

Grain size

Color

Mineral composition

Surface patterns

However, manual classification is time-consuming, subjective, and requires domain expertise.
With advancements in Deep Learning and Computer Vision, it has become possible to automate this process using image-based classification models.
3. Dataset Description

The dataset consists of 1023 labeled images across three classes:

Class	Number of Images
Coal	369
Limestone	331
Sandstone	323
Total	1023

**Characteristics of the dataset:**

Images are stored in separate class folders.

They vary in size, lighting, and camera quality.

Dataset split: 80% training, 20% validation.

Contains .jpg, .jpeg, and .png formats.

**Challenges:**

Limited number of images.

High visual similarity between limestone and sandstone.

Lighting variations affecting color and texture representation.
