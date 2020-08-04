# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Check copyright year """

import sys
import os
import datetime
import argparse
import subprocess
import traceback


class YearChecker:
    """ Check copyright year """

    def __init__(self, root_dir):
        self._root_dir = root_dir

    @staticmethod
    def _exception_to_string(excp):
        stack = traceback.extract_stack()[:-3] + traceback.extract_tb(excp.__traceback__)
        pretty = traceback.format_list(stack)
        return ''.join(pretty) + '\n  {} {}'.format(excp.__class__, excp)

    @staticmethod
    def _get_year_from_date(date):
        if not date or len(date) < 4:
            return None

        return int(date[:4])

    @staticmethod
    def _format_output(out, err):
        out = out.decode('utf-8').strip()
        err = err.decode('utf-8').strip()
        err = err if err else None
        year = YearChecker._get_year_from_date(out)
        return year, err

    def _process_file_last_year(self, relative_path):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        popen = subprocess.Popen(['git', 'log', '-1', '--format=%aI', relative_path],
                                 cwd=self._root_dir,
                                 env=env,
                                 stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        out, err = popen.communicate()
        popen.wait()
        return YearChecker._format_output(out, err)

    def _get_file_last_year(self, relative_path):
        last_year = None
        errors = []
        try:
            last_year, err = self._process_file_last_year(relative_path)
            if err:
                errors.append(err)
        except Exception as ex:  # pylint: disable=broad-except
            errors.append("'{}' Last year: {}".format(relative_path, str(ex)))

        if errors:
            raise ValueError(' - '.join(errors))

        return last_year

    def check_copyright_year(self, file_path):
        """ check copyright year for a file """
        now = datetime.datetime.now()
        file_with_invalid_year = False
        file_has_header = False
        try:
            with open(file_path, 'rt', encoding="utf8") as file:
                for line in file:
                    if not line.startswith('# (C) Copyright IBM '):
                        continue

                    file_has_header = True
                    curr_years = []
                    for word in line.strip().split():
                        for year in word.strip().split(','):
                            if year.startswith('20') and len(year) >= 4:
                                try:
                                    curr_years.append(int(year[0:4]))
                                except ValueError:
                                    pass

                    header_start_year = None
                    header_last_year = None
                    if len(curr_years) > 1:
                        header_start_year = curr_years[0]
                        header_last_year = curr_years[1]
                    elif len(curr_years) == 1:
                        header_start_year = header_last_year = curr_years[0]

                    relative_path = os.path.relpath(file_path, self._root_dir)
                    last_year = self._get_file_last_year(relative_path)
                    if last_year and header_last_year != last_year:
                        new_line = '# (C) Copyright IBM '
                        if header_start_year and header_start_year != last_year:
                            new_line += '{}, '.format(header_start_year)

                        new_line += '{}.\n'.format(now.year)
                        print("Wrong Copyright Year: '{}': Current: '{}' Correct: '{}'".format(
                            relative_path, line[:-1], new_line[:-1]))

                        file_with_invalid_year = True

                    break

        except UnicodeDecodeError:
            return False, False

        return file_with_invalid_year, file_has_header

    def check_year(self):
        """ check copyright year """
        return self._check_year(self._root_dir)

    def _check_year(self, path):
        files_with_invalid_years = 0
        files_with_header = 0
        for item in os.listdir(path):
            fullpath = os.path.join(path, item)
            if os.path.isdir(fullpath):
                if not item.startswith('.git'):
                    files = self._check_year(fullpath)
                    files_with_invalid_years += files[0]
                    files_with_header += files[1]
                continue

            if os.path.isfile(fullpath):
                file_with_invalid_year, file_has_header = self.check_copyright_year(fullpath)
                if file_with_invalid_year:
                    files_with_invalid_years += 1
                if file_has_header:
                    files_with_header += 1

        return files_with_invalid_years, files_with_header


def check_path(path):
    """ valid path argument """
    if not path or os.path.isdir(path):
        return path

    raise argparse.ArgumentTypeError("readable_dir:{} is not a valid path".format(path))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Qiskit Check Copyright Year Tool')
    PARSER.add_argument('-path',
                        type=check_path,
                        metavar='path',
                        help='Root path of project.')

    ARGS = PARSER.parse_args()
    if not ARGS.path:
        ARGS.path = os.getcwd()

    INVALID_YEARS, HAS_HEADER = YearChecker(ARGS.path).check_year()
    print("{} of {} files with copyright header have wrong years.".format(
        INVALID_YEARS, HAS_HEADER))

    sys.exit(os.EX_OK if INVALID_YEARS == 0 else os.EX_SOFTWARE)
