import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="unsupervised_depth_and_motion_estimation",
    version="0.0.1",
    author="Denis Tananaev",
    author_email="d.d.tananaev@gmail.com",
    description="The deep learning for depth and motion estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: author copyright",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
