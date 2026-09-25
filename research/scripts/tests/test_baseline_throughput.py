import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('throughput',Path(__file__).parents[1]/'analyze_baseline_throughput.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(step=i,wall_time=100+i*10,metrics={'timing_s/step':10.,'timing_s/save_checkpoint':2. if i==10 else 0.,'response_length/mean':100.}) for i in range(1,11)]
        self.events=[dict(event='end',phase='_save_checkpoint',status='complete',seconds=2.)]
    def test_no_double_count_checkpoint(self):
        result=module.summarize(self.rows,self.events)
        self.assertEqual(result['sustained_steps'],8)
        self.assertEqual(result['end_to_end_step_seconds']['mean'],10)
        self.assertEqual(result['train_step_excluding_save_validation_seconds']['mean'],9.75)
    def test_reject_short_window(self):
        with self.assertRaises(ValueError): module.summarize(self.rows[:9],self.events)
    def test_require_observed_save(self):
        with self.assertRaises(ValueError): module.summarize(self.rows,[])
    def test_reject_duplicate_steps(self):
        with self.assertRaises(ValueError): module.summarize(self.rows+self.rows[:1],self.events)
    def test_terminal_accounting_includes_external_phases_once(self):
        events=self.events+[dict(event='end',phase=phase,status='complete',seconds=seconds)
                            for phase,seconds in [('init_workers',5.),('_validate',7.),('fit',110.)]]
        result=module.summarize(self.rows,events,dict(status='complete',exit=0,elapsed_seconds=120.))
        self.assertEqual(result['accounted_seconds'],112.)
        self.assertEqual(result['other_startup_orchestration_cleanup_seconds'],8.)
        self.assertEqual(result['first_two_step_seconds'],20.)
        self.assertTrue(result['terminal_success'])
    def test_partial_run_does_not_invent_total_wall_time(self):
        result=module.summarize(self.rows,self.events,dict(status='running'))
        self.assertFalse(result['terminal_success'])
        self.assertIsNone(result['launcher_elapsed_seconds'])
        self.assertIsNone(result['other_startup_orchestration_cleanup_seconds'])
    def test_reject_missing_step(self):
        rows=[dict(r,step=r['step']+1) for r in self.rows]
        with self.assertRaises(ValueError): module.summarize(rows,self.events)
    def test_reject_impossible_phase_sum(self):
        with self.assertRaises(ValueError):
            module.summarize(self.rows,self.events,dict(status='complete',exit=0,elapsed_seconds=90.))
    def test_reject_nonfinite_timing(self):
        self.rows[-1]['metrics']['timing_s/step']=float('nan')
        with self.assertRaises(ValueError): module.summarize(self.rows,self.events)

if __name__=='__main__': unittest.main()
