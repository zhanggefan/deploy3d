from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import pathlib
import os
from distutils import sysconfig


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuildExtension(build_ext):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))

        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' +
            str(extdir.parent.absolute()),
            f'-DCMAKE_BUILD_TYPE={config}',
            f'-DPython_INCLUDE_DIRS={";".join(self.include_dirs)}',
            f'-DPython_SOABI={sysconfig.get_config_var("SOABI")}'
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['make', '-j'])
        os.chdir(str(cwd))


if __name__ == '__main__':
    setup(
        name='deploy3d',
        version='0.0.1',
        description=('deploy3d'),
        long_description='deploy3d',
        long_description_content_type='text/markdown',
        author='zhanggefan',
        author_email='gefan.zhang@cowarobot.com',
        keywords='deploy3d',
        packages=find_packages(),
        include_package_data=True,
        license='',
        ext_modules=[CMakeExtension(name='deploy3d.libs._')],
        cmdclass={'build_ext': CMakeBuildExtension},
        zip_safe=False)
