"""Pure scalar payload and signed break-even algebra tests."""
import unittest

from semcache.system_cost.common import Dimensions
from semcache.system_cost.network import transfer_ms
from semcache.diagnostics.r2d_network import break_even, modeled_point, payload_ledger, rtt_sensitivity


class PayloadTests(unittest.TestCase):
    dims = Dimensions('facebook/opt-2.7b', 2560, 32)

    def ledger(self, reused=16, boundary=True):
        return payload_ledger(self.dims, 32, reused, hidden_element_bytes=2,
                              delta_element_bytes=4, boundary_transfers=boundary)

    def test_directional_per_layer_boundaries_and_different_widths(self):
        ledger = self.ledger()
        hidden, deltas, h0, hL = ledger['entries']
        self.assertEqual(hidden['no_reuse_bytes_per_occurrence'], 32*2560*2)
        self.assertEqual(deltas['reuse_bytes_per_occurrence'], 3*16*2560*4)
        self.assertEqual(ledger['no_reuse']['ud_to_es_bytes'], 32*3*32*2560*4+32*2560*2)
        self.assertEqual(ledger['reuse']['es_to_ud_bytes'], 32*16*2560*2+32*2560*2)
        self.assertEqual(ledger['saved']['total_network_bytes'], 32*16*2560*(2+3*4))
        self.assertEqual(h0['saved_bytes'], 0)
        self.assertEqual(hL['saved_bytes'], 0)
        self.assertIsNone(hidden['application_message_count'])
        self.assertEqual(sum(e['reuse_bytes'] for e in ledger['entries']), ledger['reuse']['total_network_bytes'])

    def test_zero_and_full_reuse_keep_boundaries_exactly_once(self):
        self.assertEqual(self.ledger(0)['saved']['total_network_bytes'], 0)
        all_reuse = self.ledger(32)
        self.assertEqual(all_reuse['reuse']['total_network_bytes'], 2*32*2560*2)
        self.assertEqual(self.ledger(32,False)['reuse']['total_network_bytes'], 0)
        self.assertEqual(transfer_ms(25_000_000, 200), 1000.)
        self.assertEqual(transfer_ms(25_000_000, 1000), 200.)

    def test_explicit_hypothetical_rtt_counts_are_not_tensor_counts(self):
        for fresh in (1,16,32):
            rtt = rtt_sensitivity(32,fresh,10,boundary_transfers=True)
            self.assertEqual(rtt['no_reuse_exchanges'],33)
            self.assertEqual(rtt['reuse_exchanges'],33)
            self.assertEqual(rtt['delta_rtt_ms'],0)
            self.assertEqual(rtt['actual_rtt_status'],'NOT_ESTIMABLE')
            self.assertIsNone(rtt['actual_message_counts'])
            self.assertEqual(rtt['provenance'],'HYPOTHETICAL_RTT')
        rtt = rtt_sensitivity(32,0,10,boundary_transfers=True)
        self.assertEqual(rtt['reuse_exchanges'],1)
        self.assertEqual(rtt['delta_rtt_ms'],-320)
        custom = rtt_sensitivity(32,1,10,boundary_transfers=True,exchanges_per_layer=3,boundary_exchanges=2)
        self.assertEqual(custom['no_reuse_exchanges'],98)
        self.assertEqual(rtt_sensitivity(32,1,0,boundary_transfers=False)['actual_rtt_status'],'PAYLOAD_ONLY_ZERO_RTT')

    def test_modeled_components_and_negative_speedup_not_clipped(self):
        rtt = rtt_sensitivity(32,16,10,boundary_transfers=True)
        point = modeled_point(1,1_000,self.ledger(),1000,rtt,include_hypothetical_rtt=True)
        self.assertGreater(point['reuse_minus_no_reuse_ms'],0)
        self.assertLess(point['speedup_ratio'],1)
        self.assertLess(point['speedup_percent'],0)
        self.assertAlmostEqual(point['no_reuse_total_ms'],1+point['no_reuse_transfer_ms']+330)

    def test_invalid_values_and_bandwidth_rejected(self):
        rtt = rtt_sensitivity(32,16,0,boundary_transfers=True)
        for b in (0,-1,float('nan'),float('inf'),True):
            with self.subTest(b=b), self.assertRaises(ValueError):
                modeled_point(1,1,self.ledger(),b,rtt,include_hypothetical_rtt=False)
        for v in (-1,float('nan'),True):
            with self.assertRaises(ValueError):
                rtt_sensitivity(32,16,v,boundary_transfers=True)
        with self.assertRaises(ValueError):
            self.ledger(33)


class RootTests(unittest.TestCase):
    def test_both_tradeoff_signs_and_signed_rtt(self):
        root = break_even(4,-1000)
        self.assertEqual(root['break_even_bandwidth_mbps'],2.)
        self.assertEqual(root['winning_region'],'BELOW_ROOT')
        self.assertEqual(root['monotonic_regime'],'INCREASING')
        opposite = break_even(-4,1000)
        self.assertEqual(opposite['break_even_bandwidth_mbps'],2.)
        self.assertEqual(opposite['winning_region'],'ABOVE_ROOT')
        self.assertEqual(break_even(4,-1000,-2)['break_even_bandwidth_mbps'],4.)

    def test_no_root_always_win_lose_tied_and_zero_bytes(self):
        for offset, count in ((-1,-1),(-1,0),(0,-1)):
            self.assertEqual(break_even(offset,count)['status'],'ALWAYS_WINNING')
        for offset, count in ((1,1),(1,0),(0,1)):
            self.assertEqual(break_even(offset,count)['status'],'ALWAYS_LOSING')
        self.assertEqual(break_even(0,0)['status'],'TIED')
        self.assertIsNone(break_even(0,-1)['break_even_bandwidth_mbps'])
        for value in (float('nan'),float('inf'),True,'1'):
            with self.assertRaises(ValueError):
                break_even(value,0)


if __name__ == '__main__':
    unittest.main()
