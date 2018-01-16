import unittest
import dax
import cpf


class TestSample(unittest.TestCase):
    def setUp(self):
        self.jobs, self.parents = dax.read_file("data/paper.xml")
        self.s = cpf.CriticalPathFirst(self.jobs)

    def test_mip(self):
        self.assertEqual(self.s.most_influential_parent('V09'), 'V03')
        self.assertEqual(self.s.most_influential_parent('V10'), 'V07')
        self.assertEqual(self.s.most_influential_parent('V11'), 'V07')
        self.assertIn(self.s.most_influential_parent('V12'), ['V07', 'V06'])

    def test_eft(self):
        self.assertEqual(self.s.earliest_finish('V12'), 21)

    def test_compute(self):
        pass
        # print self.s.compute()
        # pprint(self.s.schedule_by_resource)
        # pprint(list(self.s.get_slots(1)))


if __name__ == '__main__':
    unittest.main()
