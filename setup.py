from setuptools import setup

setup(
   name='EntregaLibreriaPy',
   version='0.0.1',
   author='Diego Yáñez',
   author_email='dyanez001@ikasle.ehu.eus',
   packages=['EntregaLibreriaPy','EntregaLibreriaPy.test'],
   license='MIT',
   description='Esta librería es el entregable para el curso de Software Matemático Estadístico. Contiene funciones que permiten gestionar data sets',
   long_description=open('README.txt').read(),
   tests_require=['pytest'],
   url="https://github.com/YanezzDiego/LibreriaPy",
   install_requires=[
      "pandas >= 0.25.1",
      "matplotlib >= 3.1.1",
      "numpy >=1.17.2",
      "seaborn >= 0.9.0"
   ],
)