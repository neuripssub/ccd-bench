"""Core evaluation suites for offensive, alignment, and interpretability research."""

from .extended_probes import (
    BabelMultilingualSuite,
    BrutusProtocolSuite,
    ChannelFactorialSuite,
    CompressionCaveatSuite,
    CryptoCommitmentTrapSuite,
    DelilahRedactionSuite,
    GoodSamaritanSuite,
    JekyllInjectionSuite,
    LotBrittlenessSuite,
    MetaEvaluatorLieSuite,
    NeedleHaystackLieSuite,
    PairedDeltaProtocolSuite,
    PolicySupersessionSuite,
    ScapegoatFalsePremiseSuite,
)
from .judas_protocol import JudasProtocolSuite
from .lazarus_self_repair import LazarusSelfRepairSuite
from .polymorphic_malware_test import PolymorphicMalwareSuite
from .shutdown_paradox_test import ShutdownParadoxSuite

__all__ = [
    "BabelMultilingualSuite",
    "BrutusProtocolSuite",
    "ChannelFactorialSuite",
    "CompressionCaveatSuite",
    "CryptoCommitmentTrapSuite",
    "DelilahRedactionSuite",
    "GoodSamaritanSuite",
    "JekyllInjectionSuite",
    "JudasProtocolSuite",
    "LazarusSelfRepairSuite",
    "LotBrittlenessSuite",
    "MetaEvaluatorLieSuite",
    "NeedleHaystackLieSuite",
    "PairedDeltaProtocolSuite",
    "PolymorphicMalwareSuite",
    "PolicySupersessionSuite",
    "ScapegoatFalsePremiseSuite",
    "ShutdownParadoxSuite",
]
