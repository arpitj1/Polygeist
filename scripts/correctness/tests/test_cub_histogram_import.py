"""Histogram import proof checks initialization and update values."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/correctness'),str(ROOT/'scripts/correctness/lib_v2')]
import recipe
from region_match import match_regions

class HistogramImport(unittest.TestCase):
    def test_initialization_and_increment(self):
        for count,bins,initial,increment,expected in [(13,7,0,1,True),(99,31,0,1,True),(13,7,1,1,False),(13,7,0,2,False)]:
            with self.subTest(count=count,bins=bins,initial=initial,increment=increment),tempfile.TemporaryDirectory() as tmp:
                d=Path(tmp);source=d/'input.c'
                source.write_text(f'''void unrelated(int input[{count}],int output[{bins}]) {{
for(int b=0;b<{bins};++b)output[b]={initial};
for(int i=0;i<{count};++i)output[input[i]]+={increment};
}}''')
                raised=recipe.raise_c(recipe.PRODUCTION,str(source),'unrelated',str(d/'raised'),[],60)
                self.assertTrue(all(raised[k].ok for k in ('cgeist','raise','debuf')))
                with patch.dict(os.environ,POLYGEIST_NUMERIC_MODEL='relaxed'):
                    matches=match_regions(Path(raised['paths']['debuf']).read_text(),str(ROOT/'generic_solver/lib_v3'),allowed_specs={'cub_histogram_even_i32'})
                self.assertEqual(bool(matches),expected)
                if matches:self.assertEqual([m.spec for m in matches],['cub_histogram_even_i32'])
if __name__=='__main__':unittest.main()
