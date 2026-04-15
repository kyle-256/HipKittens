
#pragma once

#include "base_types.cuh"
#include "util.cuh"

namespace kittens {

namespace macros {

// Macro to generate clobber for a specific register number
#define CLOBBER_AREG_CASE(N) case N: asm volatile("" ::: "a" #N); break;
#define CLOBBER_VREG_CASE(N) case N: asm volatile("" ::: "v" #N); break;

template<int GPR>
__device__ __forceinline__ void clobber_gpr() {
  if constexpr (GPR >= 256) {
    constexpr int reg = GPR - 256;
    switch (reg) {
      CLOBBER_AREG_CASE(0) CLOBBER_AREG_CASE(1) CLOBBER_AREG_CASE(2) CLOBBER_AREG_CASE(3)
      CLOBBER_AREG_CASE(4) CLOBBER_AREG_CASE(5) CLOBBER_AREG_CASE(6) CLOBBER_AREG_CASE(7)
      CLOBBER_AREG_CASE(8) CLOBBER_AREG_CASE(9) CLOBBER_AREG_CASE(10) CLOBBER_AREG_CASE(11)
      CLOBBER_AREG_CASE(12) CLOBBER_AREG_CASE(13) CLOBBER_AREG_CASE(14) CLOBBER_AREG_CASE(15)
      CLOBBER_AREG_CASE(16) CLOBBER_AREG_CASE(17) CLOBBER_AREG_CASE(18) CLOBBER_AREG_CASE(19)
      CLOBBER_AREG_CASE(20) CLOBBER_AREG_CASE(21) CLOBBER_AREG_CASE(22) CLOBBER_AREG_CASE(23)
      CLOBBER_AREG_CASE(24) CLOBBER_AREG_CASE(25) CLOBBER_AREG_CASE(26) CLOBBER_AREG_CASE(27)
      CLOBBER_AREG_CASE(28) CLOBBER_AREG_CASE(29) CLOBBER_AREG_CASE(30) CLOBBER_AREG_CASE(31)
      CLOBBER_AREG_CASE(32) CLOBBER_AREG_CASE(33) CLOBBER_AREG_CASE(34) CLOBBER_AREG_CASE(35)
      CLOBBER_AREG_CASE(36) CLOBBER_AREG_CASE(37) CLOBBER_AREG_CASE(38) CLOBBER_AREG_CASE(39)
      CLOBBER_AREG_CASE(40) CLOBBER_AREG_CASE(41) CLOBBER_AREG_CASE(42) CLOBBER_AREG_CASE(43)
      CLOBBER_AREG_CASE(44) CLOBBER_AREG_CASE(45) CLOBBER_AREG_CASE(46) CLOBBER_AREG_CASE(47)
      CLOBBER_AREG_CASE(48) CLOBBER_AREG_CASE(49) CLOBBER_AREG_CASE(50) CLOBBER_AREG_CASE(51)
      CLOBBER_AREG_CASE(52) CLOBBER_AREG_CASE(53) CLOBBER_AREG_CASE(54) CLOBBER_AREG_CASE(55)
      CLOBBER_AREG_CASE(56) CLOBBER_AREG_CASE(57) CLOBBER_AREG_CASE(58) CLOBBER_AREG_CASE(59)
      CLOBBER_AREG_CASE(60) CLOBBER_AREG_CASE(61) CLOBBER_AREG_CASE(62) CLOBBER_AREG_CASE(63)
      CLOBBER_AREG_CASE(64) CLOBBER_AREG_CASE(65) CLOBBER_AREG_CASE(66) CLOBBER_AREG_CASE(67)
      CLOBBER_AREG_CASE(68) CLOBBER_AREG_CASE(69) CLOBBER_AREG_CASE(70) CLOBBER_AREG_CASE(71)
      CLOBBER_AREG_CASE(72) CLOBBER_AREG_CASE(73) CLOBBER_AREG_CASE(74) CLOBBER_AREG_CASE(75)
      CLOBBER_AREG_CASE(76) CLOBBER_AREG_CASE(77) CLOBBER_AREG_CASE(78) CLOBBER_AREG_CASE(79)
      CLOBBER_AREG_CASE(80) CLOBBER_AREG_CASE(81) CLOBBER_AREG_CASE(82) CLOBBER_AREG_CASE(83)
      CLOBBER_AREG_CASE(84) CLOBBER_AREG_CASE(85) CLOBBER_AREG_CASE(86) CLOBBER_AREG_CASE(87)
      CLOBBER_AREG_CASE(88) CLOBBER_AREG_CASE(89) CLOBBER_AREG_CASE(90) CLOBBER_AREG_CASE(91)
      CLOBBER_AREG_CASE(92) CLOBBER_AREG_CASE(93) CLOBBER_AREG_CASE(94) CLOBBER_AREG_CASE(95)
      CLOBBER_AREG_CASE(96) CLOBBER_AREG_CASE(97) CLOBBER_AREG_CASE(98) CLOBBER_AREG_CASE(99)
      CLOBBER_AREG_CASE(100) CLOBBER_AREG_CASE(101) CLOBBER_AREG_CASE(102) CLOBBER_AREG_CASE(103)
      CLOBBER_AREG_CASE(104) CLOBBER_AREG_CASE(105) CLOBBER_AREG_CASE(106) CLOBBER_AREG_CASE(107)
      CLOBBER_AREG_CASE(108) CLOBBER_AREG_CASE(109) CLOBBER_AREG_CASE(110) CLOBBER_AREG_CASE(111)
      CLOBBER_AREG_CASE(112) CLOBBER_AREG_CASE(113) CLOBBER_AREG_CASE(114) CLOBBER_AREG_CASE(115)
      CLOBBER_AREG_CASE(116) CLOBBER_AREG_CASE(117) CLOBBER_AREG_CASE(118) CLOBBER_AREG_CASE(119)
      CLOBBER_AREG_CASE(120) CLOBBER_AREG_CASE(121) CLOBBER_AREG_CASE(122) CLOBBER_AREG_CASE(123)
      CLOBBER_AREG_CASE(124) CLOBBER_AREG_CASE(125) CLOBBER_AREG_CASE(126) CLOBBER_AREG_CASE(127)
      CLOBBER_AREG_CASE(128) CLOBBER_AREG_CASE(129) CLOBBER_AREG_CASE(130) CLOBBER_AREG_CASE(131)
      CLOBBER_AREG_CASE(132) CLOBBER_AREG_CASE(133) CLOBBER_AREG_CASE(134) CLOBBER_AREG_CASE(135)
      CLOBBER_AREG_CASE(136) CLOBBER_AREG_CASE(137) CLOBBER_AREG_CASE(138) CLOBBER_AREG_CASE(139)
      CLOBBER_AREG_CASE(140) CLOBBER_AREG_CASE(141) CLOBBER_AREG_CASE(142) CLOBBER_AREG_CASE(143)
      CLOBBER_AREG_CASE(144) CLOBBER_AREG_CASE(145) CLOBBER_AREG_CASE(146) CLOBBER_AREG_CASE(147)
      CLOBBER_AREG_CASE(148) CLOBBER_AREG_CASE(149) CLOBBER_AREG_CASE(150) CLOBBER_AREG_CASE(151)
      CLOBBER_AREG_CASE(152) CLOBBER_AREG_CASE(153) CLOBBER_AREG_CASE(154) CLOBBER_AREG_CASE(155)
      CLOBBER_AREG_CASE(156) CLOBBER_AREG_CASE(157) CLOBBER_AREG_CASE(158) CLOBBER_AREG_CASE(159)
      CLOBBER_AREG_CASE(160) CLOBBER_AREG_CASE(161) CLOBBER_AREG_CASE(162) CLOBBER_AREG_CASE(163)
      CLOBBER_AREG_CASE(164) CLOBBER_AREG_CASE(165) CLOBBER_AREG_CASE(166) CLOBBER_AREG_CASE(167)
      CLOBBER_AREG_CASE(168) CLOBBER_AREG_CASE(169) CLOBBER_AREG_CASE(170) CLOBBER_AREG_CASE(171)
      CLOBBER_AREG_CASE(172) CLOBBER_AREG_CASE(173) CLOBBER_AREG_CASE(174) CLOBBER_AREG_CASE(175)
      CLOBBER_AREG_CASE(176) CLOBBER_AREG_CASE(177) CLOBBER_AREG_CASE(178) CLOBBER_AREG_CASE(179)
      CLOBBER_AREG_CASE(180) CLOBBER_AREG_CASE(181) CLOBBER_AREG_CASE(182) CLOBBER_AREG_CASE(183)
      CLOBBER_AREG_CASE(184) CLOBBER_AREG_CASE(185) CLOBBER_AREG_CASE(186) CLOBBER_AREG_CASE(187)
      CLOBBER_AREG_CASE(188) CLOBBER_AREG_CASE(189) CLOBBER_AREG_CASE(190) CLOBBER_AREG_CASE(191)
      CLOBBER_AREG_CASE(192) CLOBBER_AREG_CASE(193) CLOBBER_AREG_CASE(194) CLOBBER_AREG_CASE(195)
      CLOBBER_AREG_CASE(196) CLOBBER_AREG_CASE(197) CLOBBER_AREG_CASE(198) CLOBBER_AREG_CASE(199)
      CLOBBER_AREG_CASE(200) CLOBBER_AREG_CASE(201) CLOBBER_AREG_CASE(202) CLOBBER_AREG_CASE(203)
      CLOBBER_AREG_CASE(204) CLOBBER_AREG_CASE(205) CLOBBER_AREG_CASE(206) CLOBBER_AREG_CASE(207)
      CLOBBER_AREG_CASE(208) CLOBBER_AREG_CASE(209) CLOBBER_AREG_CASE(210) CLOBBER_AREG_CASE(211)
      CLOBBER_AREG_CASE(212) CLOBBER_AREG_CASE(213) CLOBBER_AREG_CASE(214) CLOBBER_AREG_CASE(215)
      CLOBBER_AREG_CASE(216) CLOBBER_AREG_CASE(217) CLOBBER_AREG_CASE(218) CLOBBER_AREG_CASE(219)
      CLOBBER_AREG_CASE(220) CLOBBER_AREG_CASE(221) CLOBBER_AREG_CASE(222) CLOBBER_AREG_CASE(223)
      CLOBBER_AREG_CASE(224) CLOBBER_AREG_CASE(225) CLOBBER_AREG_CASE(226) CLOBBER_AREG_CASE(227)
      CLOBBER_AREG_CASE(228) CLOBBER_AREG_CASE(229) CLOBBER_AREG_CASE(230) CLOBBER_AREG_CASE(231)
      CLOBBER_AREG_CASE(232) CLOBBER_AREG_CASE(233) CLOBBER_AREG_CASE(234) CLOBBER_AREG_CASE(235)
      CLOBBER_AREG_CASE(236) CLOBBER_AREG_CASE(237) CLOBBER_AREG_CASE(238) CLOBBER_AREG_CASE(239)
      CLOBBER_AREG_CASE(240) CLOBBER_AREG_CASE(241) CLOBBER_AREG_CASE(242) CLOBBER_AREG_CASE(243)
      CLOBBER_AREG_CASE(244) CLOBBER_AREG_CASE(245) CLOBBER_AREG_CASE(246) CLOBBER_AREG_CASE(247)
      CLOBBER_AREG_CASE(248) CLOBBER_AREG_CASE(249) CLOBBER_AREG_CASE(250) CLOBBER_AREG_CASE(251)
      CLOBBER_AREG_CASE(252) CLOBBER_AREG_CASE(253) CLOBBER_AREG_CASE(254) CLOBBER_AREG_CASE(255)
      // Add more register numbers as needed (up to 255)
    }
  } else {
    constexpr int reg = GPR;
    switch (reg) {
      CLOBBER_VREG_CASE(0) CLOBBER_VREG_CASE(1) CLOBBER_VREG_CASE(2) CLOBBER_VREG_CASE(3)
      CLOBBER_VREG_CASE(4) CLOBBER_VREG_CASE(5) CLOBBER_VREG_CASE(6) CLOBBER_VREG_CASE(7)
      CLOBBER_VREG_CASE(8) CLOBBER_VREG_CASE(9) CLOBBER_VREG_CASE(10) CLOBBER_VREG_CASE(11)
      CLOBBER_VREG_CASE(12) CLOBBER_VREG_CASE(13) CLOBBER_VREG_CASE(14) CLOBBER_VREG_CASE(15)
      CLOBBER_VREG_CASE(16) CLOBBER_VREG_CASE(17) CLOBBER_VREG_CASE(18) CLOBBER_VREG_CASE(19)
      CLOBBER_VREG_CASE(20) CLOBBER_VREG_CASE(21) CLOBBER_VREG_CASE(22) CLOBBER_VREG_CASE(23)
      CLOBBER_VREG_CASE(24) CLOBBER_VREG_CASE(25) CLOBBER_VREG_CASE(26) CLOBBER_VREG_CASE(27)
      CLOBBER_VREG_CASE(28) CLOBBER_VREG_CASE(29) CLOBBER_VREG_CASE(30) CLOBBER_VREG_CASE(31)
      CLOBBER_VREG_CASE(32) CLOBBER_VREG_CASE(33) CLOBBER_VREG_CASE(34) CLOBBER_VREG_CASE(35)
      CLOBBER_VREG_CASE(36) CLOBBER_VREG_CASE(37) CLOBBER_VREG_CASE(38) CLOBBER_VREG_CASE(39)
      CLOBBER_VREG_CASE(40) CLOBBER_VREG_CASE(41) CLOBBER_VREG_CASE(42) CLOBBER_VREG_CASE(43)
      CLOBBER_VREG_CASE(44) CLOBBER_VREG_CASE(45) CLOBBER_VREG_CASE(46) CLOBBER_VREG_CASE(47)
      CLOBBER_VREG_CASE(48) CLOBBER_VREG_CASE(49) CLOBBER_VREG_CASE(50) CLOBBER_VREG_CASE(51)
      CLOBBER_VREG_CASE(52) CLOBBER_VREG_CASE(53) CLOBBER_VREG_CASE(54) CLOBBER_VREG_CASE(55)
      CLOBBER_VREG_CASE(56) CLOBBER_VREG_CASE(57) CLOBBER_VREG_CASE(58) CLOBBER_VREG_CASE(59)
      CLOBBER_VREG_CASE(60) CLOBBER_VREG_CASE(61) CLOBBER_VREG_CASE(62) CLOBBER_VREG_CASE(63)
      CLOBBER_VREG_CASE(64) CLOBBER_VREG_CASE(65) CLOBBER_VREG_CASE(66) CLOBBER_VREG_CASE(67)
      CLOBBER_VREG_CASE(68) CLOBBER_VREG_CASE(69) CLOBBER_VREG_CASE(70) CLOBBER_VREG_CASE(71)
      CLOBBER_VREG_CASE(72) CLOBBER_VREG_CASE(73) CLOBBER_VREG_CASE(74) CLOBBER_VREG_CASE(75)
      CLOBBER_VREG_CASE(76) CLOBBER_VREG_CASE(77) CLOBBER_VREG_CASE(78) CLOBBER_VREG_CASE(79)
      CLOBBER_VREG_CASE(80) CLOBBER_VREG_CASE(81) CLOBBER_VREG_CASE(82) CLOBBER_VREG_CASE(83)
      CLOBBER_VREG_CASE(84) CLOBBER_VREG_CASE(85) CLOBBER_VREG_CASE(86) CLOBBER_VREG_CASE(87)
      CLOBBER_VREG_CASE(88) CLOBBER_VREG_CASE(89) CLOBBER_VREG_CASE(90) CLOBBER_VREG_CASE(91)
      CLOBBER_VREG_CASE(92) CLOBBER_VREG_CASE(93) CLOBBER_VREG_CASE(94) CLOBBER_VREG_CASE(95)
      CLOBBER_VREG_CASE(96) CLOBBER_VREG_CASE(97) CLOBBER_VREG_CASE(98) CLOBBER_VREG_CASE(99)
      CLOBBER_VREG_CASE(100) CLOBBER_VREG_CASE(101) CLOBBER_VREG_CASE(102) CLOBBER_VREG_CASE(103)
      CLOBBER_VREG_CASE(104) CLOBBER_VREG_CASE(105) CLOBBER_VREG_CASE(106) CLOBBER_VREG_CASE(107)
      CLOBBER_VREG_CASE(108) CLOBBER_VREG_CASE(109) CLOBBER_VREG_CASE(110) CLOBBER_VREG_CASE(111)
      CLOBBER_VREG_CASE(112) CLOBBER_VREG_CASE(113) CLOBBER_VREG_CASE(114) CLOBBER_VREG_CASE(115)
      CLOBBER_VREG_CASE(116) CLOBBER_VREG_CASE(117) CLOBBER_VREG_CASE(118) CLOBBER_VREG_CASE(119)
      CLOBBER_VREG_CASE(120) CLOBBER_VREG_CASE(121) CLOBBER_VREG_CASE(122) CLOBBER_VREG_CASE(123)
      CLOBBER_VREG_CASE(124) CLOBBER_VREG_CASE(125) CLOBBER_VREG_CASE(126) CLOBBER_VREG_CASE(127)
      CLOBBER_VREG_CASE(128) CLOBBER_VREG_CASE(129) CLOBBER_VREG_CASE(130) CLOBBER_VREG_CASE(131)
      CLOBBER_VREG_CASE(132) CLOBBER_VREG_CASE(133) CLOBBER_VREG_CASE(134) CLOBBER_VREG_CASE(135)
      CLOBBER_VREG_CASE(136) CLOBBER_VREG_CASE(137) CLOBBER_VREG_CASE(138) CLOBBER_VREG_CASE(139)
      CLOBBER_VREG_CASE(140) CLOBBER_VREG_CASE(141) CLOBBER_VREG_CASE(142) CLOBBER_VREG_CASE(143)
      CLOBBER_VREG_CASE(144) CLOBBER_VREG_CASE(145) CLOBBER_VREG_CASE(146) CLOBBER_VREG_CASE(147)
      CLOBBER_VREG_CASE(148) CLOBBER_VREG_CASE(149) CLOBBER_VREG_CASE(150) CLOBBER_VREG_CASE(151)
      CLOBBER_VREG_CASE(152) CLOBBER_VREG_CASE(153) CLOBBER_VREG_CASE(154) CLOBBER_VREG_CASE(155)
      CLOBBER_VREG_CASE(156) CLOBBER_VREG_CASE(157) CLOBBER_VREG_CASE(158) CLOBBER_VREG_CASE(159)
      CLOBBER_VREG_CASE(160) CLOBBER_VREG_CASE(161) CLOBBER_VREG_CASE(162) CLOBBER_VREG_CASE(163)
      CLOBBER_VREG_CASE(164) CLOBBER_VREG_CASE(165) CLOBBER_VREG_CASE(166) CLOBBER_VREG_CASE(167)
      CLOBBER_VREG_CASE(168) CLOBBER_VREG_CASE(169) CLOBBER_VREG_CASE(170) CLOBBER_VREG_CASE(171)
      CLOBBER_VREG_CASE(172) CLOBBER_VREG_CASE(173) CLOBBER_VREG_CASE(174) CLOBBER_VREG_CASE(175)
      CLOBBER_VREG_CASE(176) CLOBBER_VREG_CASE(177) CLOBBER_VREG_CASE(178) CLOBBER_VREG_CASE(179)
      CLOBBER_VREG_CASE(180) CLOBBER_VREG_CASE(181) CLOBBER_VREG_CASE(182) CLOBBER_VREG_CASE(183)
      CLOBBER_VREG_CASE(184) CLOBBER_VREG_CASE(185) CLOBBER_VREG_CASE(186) CLOBBER_VREG_CASE(187)
      CLOBBER_VREG_CASE(188) CLOBBER_VREG_CASE(189) CLOBBER_VREG_CASE(190) CLOBBER_VREG_CASE(191)
      CLOBBER_VREG_CASE(192) CLOBBER_VREG_CASE(193) CLOBBER_VREG_CASE(194) CLOBBER_VREG_CASE(195)
      CLOBBER_VREG_CASE(196) CLOBBER_VREG_CASE(197) CLOBBER_VREG_CASE(198) CLOBBER_VREG_CASE(199)
      CLOBBER_VREG_CASE(200) CLOBBER_VREG_CASE(201) CLOBBER_VREG_CASE(202) CLOBBER_VREG_CASE(203)
      CLOBBER_VREG_CASE(204) CLOBBER_VREG_CASE(205) CLOBBER_VREG_CASE(206) CLOBBER_VREG_CASE(207)
      CLOBBER_VREG_CASE(208) CLOBBER_VREG_CASE(209) CLOBBER_VREG_CASE(210) CLOBBER_VREG_CASE(211)
      CLOBBER_VREG_CASE(212) CLOBBER_VREG_CASE(213) CLOBBER_VREG_CASE(214) CLOBBER_VREG_CASE(215)
      CLOBBER_VREG_CASE(216) CLOBBER_VREG_CASE(217) CLOBBER_VREG_CASE(218) CLOBBER_VREG_CASE(219)
      CLOBBER_VREG_CASE(220) CLOBBER_VREG_CASE(221) CLOBBER_VREG_CASE(222) CLOBBER_VREG_CASE(223)
      CLOBBER_VREG_CASE(224) CLOBBER_VREG_CASE(225) CLOBBER_VREG_CASE(226) CLOBBER_VREG_CASE(227)
      CLOBBER_VREG_CASE(228) CLOBBER_VREG_CASE(229) CLOBBER_VREG_CASE(230) CLOBBER_VREG_CASE(231)
      CLOBBER_VREG_CASE(232) CLOBBER_VREG_CASE(233) CLOBBER_VREG_CASE(234) CLOBBER_VREG_CASE(235)
      CLOBBER_VREG_CASE(236) CLOBBER_VREG_CASE(237) CLOBBER_VREG_CASE(238) CLOBBER_VREG_CASE(239)
      CLOBBER_VREG_CASE(240) CLOBBER_VREG_CASE(241) CLOBBER_VREG_CASE(242) CLOBBER_VREG_CASE(243)
      CLOBBER_VREG_CASE(244) CLOBBER_VREG_CASE(245) CLOBBER_VREG_CASE(246) CLOBBER_VREG_CASE(247)
      CLOBBER_VREG_CASE(248) CLOBBER_VREG_CASE(249) CLOBBER_VREG_CASE(250) CLOBBER_VREG_CASE(251)
      CLOBBER_VREG_CASE(252) CLOBBER_VREG_CASE(253) CLOBBER_VREG_CASE(254) CLOBBER_VREG_CASE(255)
      // Add more register numbers as needed (up to 255)
    }
  }
}

#undef CLOBBER_AREG_CASE
#undef CLOBBER_VREG_CASE

__device__ __forceinline__ constexpr uint32_t max_ds_inst_offset()
{
  // DS ops contain 2 8-bits instruction offset.
  // For non-pk2 instructions like ds_read_b32, the 2 fields are regarded as 1.
  // For pk2 instructions like ds_read2_b32, max offset is limited by 8 bits.
  return (1u << 16) - 1;
}

__device__ __forceinline__ constexpr uint32_t max_ds_pk2_inst_offset()
{
  // DS ops contain 2 8-bits instruction offset.
  // For non-pk2 instructions like ds_read_b32, the 2 fields are regarded as a whole.
  // For pk2 instructions like ds_read2_b32, max offset is limited by 8 bits.
  return (1u << 8) - 1;
}

__device__ __forceinline__ constexpr uint32_t max_mubuf_inst_offset()
{
  // MUBUF ops contain 1 12-bits instruction offset.
  return (1u << 12) - 1;
}

template<int GPR_START>
__device__ __forceinline__ void ds_read_b32(const uint32_t smem_ptr, const int i_offset) {
  if constexpr (GPR_START >= 256) {
    uint32_t tmp;
    asm volatile("ds_read_b32 %0, %1 offset:%2" : "=v"(tmp) : "v"(smem_ptr), "i"(i_offset) : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    v_accvgpr_write_b32<GPR_START>(tmp);
  } else {
    asm volatile("ds_read_b32 v[%0], %1 offset:%2" : : "n"(GPR_START), "v"(smem_ptr), "i"(i_offset) : "memory");
  }
}

template <typename T>
__device__ __forceinline__ void ds_read_b32(T& dst, const uint32_t smem_ptr, const int i_offset) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  asm volatile("ds_read_b32 %0, %1 offset:%2"
    : "=v"(dst)
    : "v"(smem_ptr), "i"(i_offset)
    : "memory");
}

template <typename T = u32x2>
__device__ __forceinline__ T ds_read_b64(const uint32_t smem_ptr, const int i_offset) {
  static_assert(sizeof(T) == sizeof(uint32_t) * 2);
  T result;
  asm volatile("ds_read_b64 %0, %1 offset:%2"
    : "=v"(result)
    : "v"(smem_ptr), "i"(i_offset)
    : "memory");
  return result;
}

template<int GPR_START>
__device__ __forceinline__ void ds_read_b64(const uint32_t smem_ptr, const int i_offset) {
  constexpr int GPR_END = GPR_START + 1;
  if constexpr (GPR_START >= 256) {
    int2 tmp;
    asm volatile("ds_read_b64 %0, %1 offset:%2" : "=v"(tmp) : "v"(smem_ptr), "i"(i_offset) : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    v_accvgpr_write_b32<GPR_START  >(static_cast<uint32_t>(tmp.x));
    v_accvgpr_write_b32<GPR_START+1>(static_cast<uint32_t>(tmp.y));
  } else {
    asm volatile("ds_read_b64 v[%0:%1], %2 offset:%3" : : "n"(GPR_START), "n"(GPR_END), "v"(smem_ptr), "i"(i_offset) : "memory");
  }
}

template<int GPR_START>
__device__ __forceinline__ void ds_read_b64_tr_b16(const uint32_t smem_ptr, const int i_offset) {
  constexpr int GPR_END = GPR_START + 1;
  if constexpr (GPR_START >= 256) {
    asm volatile("ds_read_b64_tr_b16 a[%0:%1], %2 offset:%3"
      :
      : "n"(GPR_START - 256), "n"(GPR_END - 256), "v"(smem_ptr), "i"(i_offset)
      : "memory");
  } else {
    asm volatile("ds_read_b64_tr_b16 v[%0:%1], %2 offset:%3"
      :
      : "n"(GPR_START), "n"(GPR_END), "v"(smem_ptr), "i"(i_offset)
      : "memory");
  }
}

template<int GPR_START>
__device__ __forceinline__ void ds_read_b64_tr_b8(const uint32_t smem_ptr, const int i_offset) {
  constexpr int GPR_END = GPR_START + 1;
  if constexpr (GPR_START >= 256) {
    asm volatile("ds_read_b64_tr_b8 a[%0:%1], %2 offset:%3"
      :
      : "n"(GPR_START - 256), "n"(GPR_END - 256), "v"(smem_ptr), "i"(i_offset)
      : "memory");
  } else {
    asm volatile("ds_read_b64_tr_b8 v[%0:%1], %2 offset:%3"
      :
      : "n"(GPR_START), "n"(GPR_END), "v"(smem_ptr), "i"(i_offset)
      : "memory");
  }
}

template <typename T = u32x4>
__device__ __forceinline__ T ds_read_b128(const uint32_t smem_ptr, const int i_offset) {
  static_assert(sizeof(T) == sizeof(uint32_t) * 4);
  T result;
  asm volatile("ds_read_b128 %0, %1 offset:%2"
    : "=v"(result)
    : "v"(smem_ptr), "i"(i_offset)
    : "memory");
  return result;
}

template<int AGPR_DST, int _dummy=0>
__device__ __forceinline__ void v_accvgpr_write_b32(uint32_t val) {
  asm volatile("v_accvgpr_write_b32 a[%0], %1"
    :
    : "n"(AGPR_DST - 256), "v"(val)
    : "memory");
}

template<int GPR_START>
__device__ __forceinline__ void ds_read_b128(const uint32_t smem_ptr, const int i_offset) {
  constexpr int GPR_END = GPR_START + 3;
  if constexpr (GPR_START >= 256) {
    // LDS reads can only target VGPRs. For AGPR: read to temp VGPRs, then accvgpr_write.
    int4 tmp;
    asm volatile("ds_read_b128 %0, %1 offset:%2"
      : "=v"(tmp)
      : "v"(smem_ptr), "i"(i_offset)
      : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    v_accvgpr_write_b32<GPR_START  >(static_cast<uint32_t>(tmp.x));
    v_accvgpr_write_b32<GPR_START+1>(static_cast<uint32_t>(tmp.y));
    v_accvgpr_write_b32<GPR_START+2>(static_cast<uint32_t>(tmp.z));
    v_accvgpr_write_b32<GPR_START+3>(static_cast<uint32_t>(tmp.w));
  } else {
    asm volatile("ds_read_b128 v[%0:%1], %2 offset:%3"
      :
      : "n"(GPR_START), "n"(GPR_END), "v"(smem_ptr), "i"(i_offset)
      : "memory");
  }
}

template<int GPR_START>
__device__ __forceinline__ void ds_write_b32(const uint32_t smem_ptr, const int i_offset) {
  if constexpr (GPR_START >= 256) {
    asm volatile("ds_write_b32 %0, a[%1], offset:%2"
      :
      : "v"(smem_ptr), "n"(GPR_START - 256), "i"(i_offset)
      : "memory");
  } else {
    asm volatile("ds_write_b32 %0, v[%1], offset:%2"
      :
      : "v"(smem_ptr), "n"(GPR_START), "i"(i_offset)
      : "memory");
  }
}

template <typename T>
__device__ __forceinline__ void ds_write_b32(const T& val, const uint32_t smem_ptr, const int i_offset = 0) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  asm volatile("ds_write_b32 %0, %1 offset:%2"
    :
    : "v"(smem_ptr), "v"(val), "i"(i_offset)
    : "memory");
}

template<int GPR_START>
__device__ __forceinline__ void ds_write_b64(const uint32_t smem_ptr, const int i_offset) {
  if constexpr (GPR_START >= 256) {
    asm volatile("ds_write_b64 %0, a[%1:%2], offset:%3"
      :
      : "v"(smem_ptr), "n"(GPR_START - 256), "n"(GPR_START + 1 - 256), "i"(i_offset)
      : "memory");
  } else {
    asm volatile("ds_write_b64 %0, v[%1:%2], offset:%3"
      :
      : "v"(smem_ptr), "n"(GPR_START), "n"(GPR_START + 1), "i"(i_offset)
      : "memory");
  }
}

template <typename T>
__device__ __forceinline__ void ds_write_b64(const T& val, const uint32_t smem_ptr, const int i_offset = 0) {
  static_assert(sizeof(T) == 2 * sizeof(uint32_t));
  asm volatile("ds_write_b64 %0, %1 offset:%2"
    :
    : "v"(smem_ptr), "v"(val), "i"(i_offset)
    : "memory");
}

template<int GPR_START>
__device__ __forceinline__ void ds_write_b128(const uint32_t smem_ptr, const int i_offset = 0) {
  if constexpr (GPR_START >= 256) {
    asm volatile("ds_write_b128 %0, a[%1:%2], offset:%3"
      :
      : "v"(smem_ptr), "n"(GPR_START - 256), "n"(GPR_START + 3 - 256), "i"(i_offset)
      : "memory");
  } else {
    asm volatile("ds_write_b128 %0, v[%1:%2], offset:%3"
      :
      : "v"(smem_ptr), "n"(GPR_START), "n"(GPR_START + 3), "i"(i_offset)
      : "memory");
  }
}

template<typename T>
__device__ __forceinline__ void ds_write_b128(const T& value, const uint32_t smem_ptr, const int i_offset = 0) {
  static_assert(sizeof(T) == sizeof(u32x4));
  asm volatile("ds_write_b128 %0, %1 offset:%2"
    :
    : "v"(smem_ptr), "v"(value), "i"(i_offset)
    : "memory");
}

template<int GPR_START>
__device__ __forceinline__ void buffer_load_dword(const buffer_resource& br, const uint32_t v_offset, const uint32_t s_offset = 0, const int i_offset = 0) {
  if constexpr (GPR_START >= 256) {
    asm volatile("buffer_load_dword a[%0], %1, %2, %3 offen offset:%4"
      :
      : "n"(GPR_START - 256), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
      : "memory");
  } else {
    asm volatile("buffer_load_dword v[%0], %1, %2, %3 offen offset:%4"
      :
      : "n"(GPR_START), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
      : "memory");
  }
}

template<typename T = uint32_t>
__device__ __forceinline__ T buffer_load_dword(
  const buffer_resource& br, const uint32_t v_offset, const uint32_t s_offset = 0, const uint32_t i_offset = 0) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  T result;
  asm volatile("buffer_load_dword %0, %1, %2, %3 offen offset:%4"
    : "=v"(result)
    : "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
    : "memory");
  return result;
}

template<int GPR_START>
__device__ __forceinline__ void buffer_load_dwordx2(const buffer_resource& br, const uint32_t v_offset, const uint32_t s_offset = 0, const int i_offset = 0) {
  if constexpr (GPR_START >= 256) {
    asm volatile("buffer_load_dwordx2 a[%0:%1], %2, %3, %4 offen offset:%5"
      :
      : "n"(GPR_START - 256), "n"(GPR_START + 1 - 256), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
      : "memory");
  } else {
    asm volatile("buffer_load_dwordx2 v[%0:%1], %2, %3, %4 offen offset:%5"
      :
      : "n"(GPR_START), "n"(GPR_START + 1), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
      : "memory");
  }
}

template<typename T = u32x2>
__device__ __forceinline__ T buffer_load_dwordx2(
  const buffer_resource& br, const uint32_t v_offset, const uint32_t s_offset = 0, const uint32_t i_offset = 0) {
  static_assert(sizeof(T) == sizeof(uint32_t) * 2);
  T result;
  asm volatile("buffer_load_dwordx2 %0, %1, %2, %3 offen offset:%4"
    : "=v"(result)
    : "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
    : "memory");
  return result;
}

template<int GPR_START>
__device__ __forceinline__ void buffer_load_dwordx4(const buffer_resource& br, const uint32_t v_offset, const uint32_t s_offset = 0, const int i_offset = 0) {
  if constexpr (GPR_START >= 256) {
    asm volatile("buffer_load_dwordx4 a[%0:%1], %2, %3, %4 offen offset:%5"
      :
      : "n"(GPR_START - 256), "n"(GPR_START + 3 - 256), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
      : "memory");
  } else {
    asm volatile("buffer_load_dwordx4 v[%0:%1], %2, %3, %4 offen offset:%5"
      :
      : "n"(GPR_START), "n"(GPR_START + 3), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
      : "memory");
  }
}

template<typename T = u32x4>
__device__ __forceinline__ T buffer_load_dwordx4(
  const buffer_resource& br, const uint32_t v_offset, const uint32_t s_offset = 0, const uint32_t i_offset = 0) {
  static_assert(sizeof(T) == sizeof(uint32_t) * 4);
  T result;
  asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen offset:%4"
    : "=v"(result)
    : "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
    : "memory");
  return result;
}

template<int GPR>
__device__ __forceinline__ void buffer_store_dword(const buffer_resource& br, const uint32_t v_offset, const uint32_t s_offset = 0, const int i_offset = 0) {
  // AGPRS
  if constexpr (GPR >= 256) {
    asm volatile("buffer_store_dword a[%0], %1, %2, %3 offen offset:%4"
      :
      : "n"(GPR - 256), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
      : "memory");
  // VGPRS
  } else {
    asm volatile("buffer_store_dword v[%0], %1, %2, %3 offen offset:%4"
      :
      : "n"(GPR), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
      : "memory");
  }
}

template<typename T = u32x2>
__device__ __forceinline__ void buffer_store_dword(
  const T& value, const buffer_resource& br, const uint32_t v_offset, const uint32_t s_offset = 0, const uint32_t i_offset = 0) {
  static_assert(sizeof(T) == sizeof(uint32_t) * 2);
  asm volatile("buffer_store_dword %0, %1, %2, %3 offen offset:%4"
    :
    : "v"(value), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
    : "memory");
}

template<int GPR_START>
__device__ __forceinline__ void buffer_store_dwordx2(const buffer_resource& br, const uint32_t v_offset, const uint32_t s_offset = 0, const int i_offset = 0) {
  // AGPRS
  if constexpr (GPR_START >= 256) {
    asm volatile("buffer_store_dwordx2 a[%0:%1], %2, %3, %4 offen offset:%5"
      :
      : "n"(GPR_START - 256), "n"(GPR_START + 1 - 256), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
      : "memory");
  // VGPRS
  } else {
    asm volatile("buffer_store_dwordx2 v[%0:%1], %2, %3, %4 offen offset:%5"
      :
      : "n"(GPR_START), "n"(GPR_START + 1), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
      : "memory");
  }
}

template<typename T = u32x2>
__device__ __forceinline__ void buffer_store_dwordx2(
  const T& value, const buffer_resource& br, const uint32_t v_offset, const uint32_t s_offset = 0, const uint32_t i_offset = 0) {
  static_assert(sizeof(T) == sizeof(uint32_t) * 2);
  asm volatile("buffer_store_dwordx2 %0, %1, %2, %3 offen offset:%4"
    :
    : "v"(value), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
    : "memory");
}

template<int GPR_START>
__device__ __forceinline__ void buffer_store_dwordx4(const buffer_resource& br, const uint32_t v_offset, const uint32_t s_offset = 0, const int i_offset = 0) {
  // AGPRS
  if constexpr (GPR_START >= 256) {
    asm volatile("buffer_store_dwordx4 a[%0:%1], %2, %3, %4 offen offset:%5"
      :
      : "n"(GPR_START - 256), "n"(GPR_START + 3 - 256), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
      : "memory");
  // VGPRS
  } else {
    asm volatile("buffer_store_dwordx4 v[%0:%1], %2, %3, %4 offen offset:%5"
      :
      : "n"(GPR_START), "n"(GPR_START + 3), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
      : "memory");
  }
}

template<typename T = u32x4>
__device__ __forceinline__ void buffer_store_dwordx4(
  const T& value, const buffer_resource& br, const uint32_t v_offset, const uint32_t s_offset = 0, const uint32_t i_offset = 0) {
  static_assert(sizeof(T) == sizeof(uint32_t) * 4);
  asm volatile("buffer_store_dwordx4 %0, %1, %2, %3 offen offset:%4"
    :
    : "v"(value), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
    : "memory");
}

template<int GPR>
__device__ __forceinline__ void buffer_atomic_pk_add_bf16(const buffer_resource& br, const uint32_t v_offset, const uint32_t s_offset = 0, const int i_offset = 0) {
  if constexpr (GPR >= 256) {
    asm volatile("buffer_atomic_pk_add_bf16 a[%0], %1, %2, %3 offen offset:%4"
      :
      : "n"(GPR - 256), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
      : "memory");
  } else {
    asm volatile("buffer_atomic_pk_add_bf16 v[%0], %1, %2, %3 offen offset:%4"
      :
      : "n"(GPR), "v"(v_offset), "s"(*(const i32x4*)&br), "s"(s_offset), "i"(i_offset)
      : "memory");
  }
}

template<int GPR_START_A, int GPR_START_B>
__device__ __forceinline__ void mfma_f32_16x16x32_bf16(float4& D, const float4& C) {

  if constexpr (GPR_START_A >= 256 && GPR_START_B >= 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 %0, a[%1:%2], a[%3:%4], 0"
      : "=v"(D)
      : "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256));
  } else if constexpr (GPR_START_A < 256 && GPR_START_B >= 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 %0, v[%1:%2], a[%3:%4], 0"
      : "=v"(D)
      : "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256));
  } else if constexpr (GPR_START_A >= 256 && GPR_START_B < 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 %0, a[%1:%2], v[%3:%4], 0"
      : "=v"(D)
      : "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 3));
  } else {
    asm volatile("v_mfma_f32_16x16x32_bf16 %0, v[%1:%2], v[%3:%4], 0"
      : "=v"(D)
      : "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B), "n"(GPR_START_B + 3));
  }
}

template<int GPR_START_A, int GPR_START_B, int GPR_START_C, int GPR_START_D>
__device__ __forceinline__ void mfma_f32_16x16x32_bf16() {
  if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B >= 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 a[%0:%1], a[%2:%3], a[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B >= 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 a[%0:%1], a[%2:%3], a[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B < 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 a[%0:%1], a[%2:%3], v[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B >= 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 a[%0:%1], v[%2:%3], a[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B >= 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 v[%0:%1], a[%2:%3], a[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B >= 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 v[%0:%1], a[%2:%3], a[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B < 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 v[%0:%1], a[%2:%3], v[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A < 256 && GPR_START_B >= 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 v[%0:%1], v[%2:%3], a[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B >= 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 a[%0:%1], v[%2:%3], a[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B < 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 a[%0:%1], v[%2:%3], v[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B < 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 a[%0:%1], a[%2:%3], v[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B < 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 v[%0:%1], a[%2:%3], v[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A < 256 && GPR_START_B >= 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 v[%0:%1], v[%2:%3], a[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A < 256 && GPR_START_B < 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 v[%0:%1], v[%2:%3], v[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B < 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 a[%0:%1], v[%2:%3], v[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  } else {
    asm volatile("v_mfma_f32_16x16x32_bf16 v[%0:%1], v[%2:%3], v[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  }
}

template<int GPR_START_A, int GPR_START_B, int GPR_START_C, int GPR_START_D>
__device__ __forceinline__ void mfma_f32_32x32x16_bf16() {
  if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B >= 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 a[%0:%1], a[%2:%3], a[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 15 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C - 256), "n"(GPR_START_C + 15 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B >= 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 a[%0:%1], a[%2:%3], a[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 15 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C), "n"(GPR_START_C + 15));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B < 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 a[%0:%1], a[%2:%3], v[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 15 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C - 256), "n"(GPR_START_C + 15 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B >= 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 a[%0:%1], v[%2:%3], a[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 15 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C - 256), "n"(GPR_START_C + 15 - 256));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B >= 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 v[%0:%1], a[%2:%3], a[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 15), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C - 256), "n"(GPR_START_C + 15 - 256));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B >= 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 v[%0:%1], a[%2:%3], a[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 15), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C), "n"(GPR_START_C + 15));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B < 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 v[%0:%1], a[%2:%3], v[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 15), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C - 256), "n"(GPR_START_C + 15 - 256));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A < 256 && GPR_START_B >= 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 v[%0:%1], v[%2:%3], a[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 15), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C - 256), "n"(GPR_START_C + 15 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B >= 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 a[%0:%1], v[%2:%3], a[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 15 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C), "n"(GPR_START_C + 15));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B < 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 a[%0:%1], v[%2:%3], v[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 15 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C - 256), "n"(GPR_START_C + 15 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B < 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 a[%0:%1], a[%2:%3], v[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 15 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C), "n"(GPR_START_C + 15));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B < 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 v[%0:%1], a[%2:%3], v[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 15), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C), "n"(GPR_START_C + 15));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A < 256 && GPR_START_B >= 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 v[%0:%1], v[%2:%3], a[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 15), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256), "n"(GPR_START_C), "n"(GPR_START_C + 15));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A < 256 && GPR_START_B < 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 v[%0:%1], v[%2:%3], v[%4:%5], a[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 15), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C - 256), "n"(GPR_START_C + 15 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B < 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 a[%0:%1], v[%2:%3], v[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 15 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C), "n"(GPR_START_C + 15));
  } else {
    asm volatile("v_mfma_f32_32x32x16_bf16 v[%0:%1], v[%2:%3], v[%4:%5], v[%6:%7]"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 15), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B), "n"(GPR_START_B + 3), "n"(GPR_START_C), "n"(GPR_START_C + 15));
  }
}

template<int GPR_START_A, int GPR_START_B, int GPR_START_C, int GPR_START_D>
__device__ __forceinline__ void mfma_f32_16x16x32_fp8_fp8() {
  if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B >= 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 a[%0:%1], a[%2:%3], a[%4:%5], a[%6:%7]"
      :
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 1 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 1 - 256), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B >= 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 a[%0:%1], a[%2:%3], a[%4:%5], v[%6:%7]"
      :
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 1 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 1 - 256), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B < 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 a[%0:%1], a[%2:%3], v[%4:%5], a[%6:%7]"
      :
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 1 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 1), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B >= 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 a[%0:%1], v[%2:%3], a[%4:%5], a[%6:%7]"
      :
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 1), "n"(GPR_START_B - 256), "n"(GPR_START_B + 1 - 256), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B >= 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 v[%0:%1], a[%2:%3], a[%4:%5], a[%6:%7]"
      :
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A - 256), "n"(GPR_START_A + 1 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 1 - 256), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B >= 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 v[%0:%1], a[%2:%3], a[%4:%5], v[%6:%7]"
      :
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A - 256), "n"(GPR_START_A + 1 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 1 - 256), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B < 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 v[%0:%1], a[%2:%3], v[%4:%5], a[%6:%7]"
      :
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A - 256), "n"(GPR_START_A + 1 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 1), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A < 256 && GPR_START_B >= 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 v[%0:%1], v[%2:%3], a[%4:%5], a[%6:%7]"
      :
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A), "n"(GPR_START_A + 1), "n"(GPR_START_B - 256), "n"(GPR_START_B + 1 - 256), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B >= 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 a[%0:%1], v[%2:%3], a[%4:%5], v[%6:%7]"
      :
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 1), "n"(GPR_START_B - 256), "n"(GPR_START_B + 1 - 256), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B < 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 a[%0:%1], v[%2:%3], v[%4:%5], a[%6:%7]"
      :
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 1), "n"(GPR_START_B), "n"(GPR_START_B + 1), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B < 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 a[%0:%1], a[%2:%3], v[%4:%5], v[%6:%7]"
      :
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 1 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 1), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B < 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 v[%0:%1], a[%2:%3], v[%4:%5], v[%6:%7]"
      :
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A - 256), "n"(GPR_START_A + 1 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 1), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A < 256 && GPR_START_B >= 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 v[%0:%1], v[%2:%3], a[%4:%5], v[%6:%7]"
      :
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A), "n"(GPR_START_A + 1), "n"(GPR_START_B - 256), "n"(GPR_START_B + 1 - 256), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A < 256 && GPR_START_B < 256 && GPR_START_C >= 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 v[%0:%1], v[%2:%3], v[%4:%5], a[%6:%7]"
      :
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A), "n"(GPR_START_A + 1), "n"(GPR_START_B), "n"(GPR_START_B + 1), "n"(GPR_START_C - 256), "n"(GPR_START_C + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B < 256 && GPR_START_C < 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 a[%0:%1], v[%2:%3], v[%4:%5], v[%6:%7]"
      :
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 1), "n"(GPR_START_B), "n"(GPR_START_B + 1), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  } else {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 v[%0:%1], v[%2:%3], v[%4:%5], v[%6:%7]"
      :
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A), "n"(GPR_START_A + 1), "n"(GPR_START_B), "n"(GPR_START_B + 1), "n"(GPR_START_C), "n"(GPR_START_C + 3));
  }
}

template<int GPR_START_A, int GPR_START_B, int GPR_START_D>
__device__ __forceinline__ void mfma_f32_16x16x32_bf16_zero_accum() {
  if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B >= 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 a[%0:%1], a[%2:%3], a[%4:%5], 0"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B >= 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 v[%0:%1], a[%2:%3], a[%4:%5], 0"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B >= 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 a[%0:%1], v[%2:%3], a[%4:%5], 0"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B < 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 a[%0:%1], a[%2:%3], v[%4:%5], 0"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 3));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B < 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 v[%0:%1], a[%2:%3], v[%4:%5], 0"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 3));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A < 256 && GPR_START_B >= 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 v[%0:%1], v[%2:%3], a[%4:%5], 0"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B < 256) {
    asm volatile("v_mfma_f32_16x16x32_bf16 a[%0:%1], v[%2:%3], v[%4:%5], 0"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B), "n"(GPR_START_B + 3));
  } else {
    asm volatile("v_mfma_f32_16x16x32_bf16 v[%0:%1], v[%2:%3], v[%4:%5], 0"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B), "n"(GPR_START_B + 3));
  }
}

template<int GPR_START_A, int GPR_START_B, int GPR_START_D>
__device__ __forceinline__ void mfma_f32_32x32x16_bf16_zero_accum() {
  if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B >= 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 a[%0:%1], a[%2:%3], a[%4:%5], 0"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 15 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B >= 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 v[%0:%1], a[%2:%3], a[%4:%5], 0"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 15), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B >= 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 a[%0:%1], v[%2:%3], a[%4:%5], 0"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 15 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B < 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 a[%0:%1], a[%2:%3], v[%4:%5], 0"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 15 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 3));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B < 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 v[%0:%1], a[%2:%3], v[%4:%5], 0"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 15), "n"(GPR_START_A - 256), "n"(GPR_START_A + 3 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 3));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A < 256 && GPR_START_B >= 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 v[%0:%1], v[%2:%3], a[%4:%5], 0"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 15), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B - 256), "n"(GPR_START_B + 3 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B < 256) {
    asm volatile("v_mfma_f32_32x32x16_bf16 a[%0:%1], v[%2:%3], v[%4:%5], 0"
      : 
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 15 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B), "n"(GPR_START_B + 3));
  } else {
    asm volatile("v_mfma_f32_32x32x16_bf16 v[%0:%1], v[%2:%3], v[%4:%5], 0"
      : 
      : "n"(GPR_START_D), "n"(GPR_START_D + 15), "n"(GPR_START_A), "n"(GPR_START_A + 3), "n"(GPR_START_B), "n"(GPR_START_B + 3));
  }
}

template<int GPR_START_A, int GPR_START_B, int GPR_START_D>
__device__ __forceinline__ void mfma_f32_16x16x32_fp8_fp8_zero_accum() {
  if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B >= 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 a[%0:%1], a[%2:%3], a[%4:%5], 0"
      :
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 1 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 1 - 256));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B >= 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 v[%0:%1], a[%2:%3], a[%4:%5], 0"
      :
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A - 256), "n"(GPR_START_A + 1 - 256), "n"(GPR_START_B - 256), "n"(GPR_START_B + 1 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B >= 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 a[%0:%1], v[%2:%3], a[%4:%5], 0"
      :
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 1), "n"(GPR_START_B - 256), "n"(GPR_START_B + 1 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A >= 256 && GPR_START_B < 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 a[%0:%1], a[%2:%3], v[%4:%5], 0"
      :
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A - 256), "n"(GPR_START_A + 1 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 1));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A >= 256 && GPR_START_B < 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 v[%0:%1], a[%2:%3], v[%4:%5], 0"
      :
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A - 256), "n"(GPR_START_A + 1 - 256), "n"(GPR_START_B), "n"(GPR_START_B + 1));
  } else if constexpr (GPR_START_D < 256 && GPR_START_A < 256 && GPR_START_B >= 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 v[%0:%1], v[%2:%3], a[%4:%5], 0"
      :
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A), "n"(GPR_START_A + 1), "n"(GPR_START_B - 256), "n"(GPR_START_B + 1 - 256));
  } else if constexpr (GPR_START_D >= 256 && GPR_START_A < 256 && GPR_START_B < 256) {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 a[%0:%1], v[%2:%3], v[%4:%5], 0"
      :
      : "n"(GPR_START_D - 256), "n"(GPR_START_D + 3 - 256), "n"(GPR_START_A), "n"(GPR_START_A + 1), "n"(GPR_START_B), "n"(GPR_START_B + 1));
  } else {
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 v[%0:%1], v[%2:%3], v[%4:%5], 0"
      :
      : "n"(GPR_START_D), "n"(GPR_START_D + 3), "n"(GPR_START_A), "n"(GPR_START_A + 1), "n"(GPR_START_B), "n"(GPR_START_B + 1));
  }
}

template<int GPR0_START, int GPR1_START, int GPR>
__device__ __forceinline__ void v_subrev_f32_dpp() {

  if constexpr (GPR0_START + 3 < 256 && GPR1_START + 3 < 256 && GPR < 256) {
    asm volatile("v_subrev_f32_dpp v[%0], v[%1], v[%2] quad_perm:[0, 0, 0, 0] row_mask:0xf bank_mask:0xf"
      : 
      : "n"(GPR0_START), "n"(GPR), "n"(GPR1_START));
    asm volatile("v_subrev_f32_dpp v[%0], v[%1], v[%2] quad_perm:[1, 1, 1, 1] row_mask:0xf bank_mask:0xf"
      : 
      : "n"(GPR0_START + 1), "n"(GPR), "n"(GPR1_START + 1));
    asm volatile("v_subrev_f32_dpp v[%0], v[%1], v[%2] quad_perm:[2, 2, 2, 2] row_mask:0xf bank_mask:0xf"
      : 
      : "n"(GPR0_START + 2), "n"(GPR), "n"(GPR1_START + 2));
    asm volatile("v_subrev_f32_dpp v[%0], v[%1], v[%2] quad_perm:[3, 3, 3, 3] row_mask:0xf bank_mask:0xf"
      : 
      : "n"(GPR0_START + 3), "n"(GPR), "n"(GPR1_START + 3));
  } else {
    static_assert(false, "Invalid operand for instruction: v_subrev_f32_dpp");
  }
}

template<int DST_GPR, int SRC_GPR_0, int SRC_GPR_1>
__device__ __forceinline__ void v_cvt_pk_bf16_f32() {
  if constexpr (DST_GPR < 256 && SRC_GPR_0 < 256 && SRC_GPR_1 < 256) {
    asm volatile("v_cvt_pk_bf16_f32 v[%0], v[%1], v[%2]"
      : 
      : "n"(DST_GPR), "n"(SRC_GPR_0), "n"(SRC_GPR_1));
  } else {
    static_assert(false, "Invalid operand for instruction: v_cvt_pk_bf16_f32");
  }
}

template<int GPR0, int GPR1>
__device__ __forceinline__ void v_permlane16_swap_b32_e32() {
  if constexpr (GPR0 < 256 && GPR1 < 256) {
    asm volatile("v_permlane16_swap_b32_e32 v[%0], v[%1]"
      : 
      : "n"(GPR0), "n"(GPR1));
  } else {
    static_assert(false, "Invalid operand for instruction: v_permlane16_swap_b32_e32");
  }
}

template<int GPR0, int GPR1>
__device__ __forceinline__ void v_accvgpr_read_b32() {
  asm volatile("v_accvgpr_read_b32 v[%0], a[%1]"
    : 
    : "n"(GPR0), "n"(GPR1 - 256)
    : "memory");
}

template<int AGPR_DST, int VGPR_SRC>
__device__ __forceinline__ void v_accvgpr_write_b32() {
  asm volatile("v_accvgpr_write_b32 a[%0], v[%1]"
    :
    : "n"(AGPR_DST - 256), "n"(VGPR_SRC)
    : "memory");
}


template<int GPR, typename T>
__device__ __forceinline__ void v_mov_b32_up2p(const T value) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  asm volatile("v_mov_b32 v[%0], %1"
    : 
    : "n"(GPR), "v"(value));
}

template <int GPR, typename T = uint32_t>
__device__ __forceinline__ T v_mov_b32_p2up() {
  static_assert(sizeof(T) == sizeof(uint32_t));
  T r;
  if constexpr (GPR < 256) {
    asm volatile("v_mov_b32 %0, v[%1]"
      : "=v"(r)
      : "n"(GPR));
  }
  else {
    asm volatile("v_accvgpr_read_b32 %0, a[%1]"
      : "=v"(r)
      : "n"(GPR - 256));
  }
  return r;
}

template<int GPR0, int GPR1>
__device__ __forceinline__ void v_mov_b32_e32() {
  asm volatile("v_mov_b32_e32 v[%0], v[%1]"
    : 
    : "n"(GPR0), "n"(GPR1));
}

template<int GPR0, int GPR1, int GPR2>
__device__ __forceinline__ void v_cndmask_b32_e64(uint64_t mask) {
  asm volatile("v_cndmask_b32_e64 v[%0], v[%1], v[%2], %3"
    : 
    : "n"(GPR0), "n"(GPR1), "n"(GPR2), "s"(mask));
}

/**
 * @brief Multiplication operation on explicit registers and immediate operand.
 */
struct mul {
  template<int GPR0, int GPR1> 
  static __device__ inline void op(const float &param) {
    const uint32_t hex = *reinterpret_cast<const uint32_t*>(&param);
    if constexpr (GPR0 < 256 && GPR1 < 256) {
      asm volatile("v_mul_f32_e32 v[%0], %2, v[%1]"
        : 
        : "n"(GPR0), "n"(GPR1), "i"(hex));
    } else {
      static_assert(false, "Invalid operand for instruction: v_mul_f32_e32");
    }
  }

  template<int GPR0, int GPR1> 
  static __device__ inline void op_pk2(const float &param) {
    op<GPR0, GPR1>(param);
    op<GPR0 + 1, GPR1 + 1>(param);
  }

  template<int GPR0, int GPR1, int GPR2>
  static __device__ inline void op() {
    if constexpr (GPR0 < 256 && GPR1 < 256 && GPR2 < 256) {
      asm volatile("v_mul_f32_e32 v[%0], v[%2], v[%1]"
        : 
        : "n"(GPR0), "n"(GPR1), "n"(GPR2));
    } else {
      static_assert(false, "Invalid operand for instruction: v_mul_f32_e32");
    }
  }

  template<int GPR0, int GPR1, int GPR2>
  static __device__ inline void op_pk2() {
    if constexpr (GPR0 < (256 - 1) && GPR1 < (256 - 1) && GPR2 < (256 - 1)) {
      asm volatile("v_pk_mul_f32 v[%0:%1], v[%4:%5], v[%2:%3]"
        : 
        : "n"(GPR0), "n"(GPR0 + 1), "n"(GPR1), "n"(GPR1 + 1), "n"(GPR2), "n"(GPR2 + 1));
    } else {
      static_assert(false, "Invalid operand for instruction: v_pk_mul_f32");
    }
  }
}; 

 struct mul_vgpr {
  template<int GPR0, int GPR1> 
  static __device__ inline void op(const float &param) {
    if constexpr (GPR0 < 256 && GPR1 < 256) {
      asm volatile("v_mul_f32_e32 v[%0], %2, v[%1]"
        : 
        : "n"(GPR0), "n"(GPR1), "v"(param));
    } else {
      static_assert(false, "Invalid operand for instruction: v_mul_f32_e32");
    }
  }

  template<int GPR0, int GPR1> 
  static __device__ inline void op_pk2(const float &param) {
    if constexpr (GPR0 < (256 - 1) && GPR1 < (256 - 1)) {
      const float2 param2 = {param, param};
      asm volatile("v_pk_mul_f32 v[%0:%1], %4, v[%2:%3]"
        : 
        : "n"(GPR0), "n"(GPR0 + 1), "n"(GPR1), "n"(GPR1 + 1), "v"(param2));
    } else {
      static_assert(false, "Invalid operand for instruction: v_pk_mul_f32");
    }
  }
};

struct exp2 {
  template<int GPR0, int GPR1> 
  static __device__ inline void op() {
    if constexpr (GPR0 < 256 && GPR1 < 256) {
      asm volatile(
        "v_exp_f32_e32 v[%0], v[%1]"
        : 
        : "n"(GPR0), "n"(GPR1));
    } else {
      static_assert(false, "Invalid operand for instruction: exp2");
    }
  }
}; 

struct zero {
  template<int GPR0, int GPR1>
  static __device__ inline void op() {
    static_assert(GPR0 == GPR1, "GPR0 and GPR1 must be the same");
    if constexpr (GPR0 < 256) {
      asm volatile("v_mov_b32 v[%0], 0"
        : 
        : "n"(GPR0));
    } else {
      asm volatile("v_accvgpr_write_b32 a[%0], 0"
        :
        : "n"(GPR0 - 256));
    }
  }
};

// ═══════════ MXFP4 MFMA with block scaling ═══════════
// v_mfma_scale_f32_16x16x128_f8f6f4 D[0:3], A[0:3], B[0:3], C[0:3], scaleA, scaleB
//   op_sel:[SEL_A, SEL_B, 0] op_sel_hi:[HI_A, HI_B, 0] cbsz:4 blgp:4
//
// Template params:
//   GPR_D/A/B/C: register start (0-255=VGPR, 256-511=AGPR)
//   GPR_SA/SB: scale registers (must be VGPR, <256)
//   SEL_A/SEL_B: op_sel bits (0 or 1, selects A/B subgroup within KPAIR)
//   HI_A/HI_B: op_sel_hi bits (0 or 1, selects K-phase lo/hi)

// Helper: generate v[N:N+3] or a[N:N+3] string prefix
#define _MFMA_FP4_REG(GPR, IDX) \
    (GPR >= 256 ? "a" : "v"), "n"(GPR >= 256 ? GPR - 256 + IDX : GPR + IDX)

// Main macro: dispatches D/A/B/C to v[] or a[] based on GPR values
// We generate all 16 combinations via a helper that builds the asm string
template<int GPR_D, int GPR_A, int GPR_B, int GPR_C, int GPR_SA, int GPR_SB,
         int SEL_A=0, int SEL_B=0, int HI_A=0, int HI_B=0>
__device__ __forceinline__ void mfma_scale_f32_16x16x128_fp4() {
    static_assert(GPR_SA < 256 && GPR_SB < 256, "Scale registers must be VGPRs");
    // Build op_sel and op_sel_hi strings at compile time via if constexpr
    // The instruction: v_mfma_scale_f32_16x16x128_f8f6f4 D, A, B, C, SA, SB [modifiers]

    // We need the D register prefix/range, A prefix/range, etc.
    // Using compile-time dispatch for D/A/B/C v/a prefixes

    constexpr auto d_lo = GPR_D >= 256 ? GPR_D - 256 : GPR_D;
    constexpr auto d_hi = d_lo + 3;
    constexpr auto a_lo = GPR_A >= 256 ? GPR_A - 256 : GPR_A;
    constexpr auto a_hi = a_lo + 3;
    constexpr auto b_lo = GPR_B >= 256 ? GPR_B - 256 : GPR_B;
    constexpr auto b_hi = b_lo + 3;
    constexpr auto c_lo = GPR_C >= 256 ? GPR_C - 256 : GPR_C;
    constexpr auto c_hi = c_lo + 3;

    // Most common case for GEMM: D=AGPR, A=VGPR, B=VGPR/AGPR, C=AGPR
    // Generate op_sel/op_sel_hi modifier string
    // op_sel:[SEL_A, SEL_B, 0]  op_sel_hi:[HI_A, HI_B, 0]

    // For simplicity and correctness, enumerate all op_sel/phase combos
    // The instruction encoding changes based on op_sel bits

    // Phase 0 (op_sel_hi:[0,0,0]): first 128 FP4 of K-step
    // Phase 1 (op_sel_hi:[1,1,0]): second 128 FP4 of K-step

    #define _FP4_ASM_BODY(D_PFX, A_PFX, B_PFX, C_PFX, OPSEL, OPSELHI) \
        asm volatile("v_mfma_scale_f32_16x16x128_f8f6f4 " \
            D_PFX "[%0:%1], " A_PFX "[%2:%3], " B_PFX "[%4:%5], " \
            C_PFX "[%6:%7], v[%8], v[%9] " OPSEL " " OPSELHI " cbsz:4 blgp:4" \
            : : "n"(d_lo), "n"(d_hi), "n"(a_lo), "n"(a_hi), \
                "n"(b_lo), "n"(b_hi), "n"(c_lo), "n"(c_hi), \
                "n"(GPR_SA), "n"(GPR_SB))

    // Select op_sel string
    #define _OPSEL_STR \
        (SEL_A == 0 && SEL_B == 0) ? "" : \
        (SEL_A == 0 && SEL_B == 1) ? "op_sel:[0,1,0]" : \
        (SEL_A == 1 && SEL_B == 0) ? "op_sel:[1,0,0]" : \
                                     "op_sel:[1,1,0]"
    // Can't use ternary for asm strings. Use if constexpr instead.

    // Combine: D/A/B/C prefix × op_sel × op_sel_hi
    // For maximum clarity, spell out the 4 most common D/A/B/C patterns:

    constexpr bool dA = GPR_D >= 256, aA = GPR_A >= 256, bA = GPR_B >= 256, cA = GPR_C >= 256;

    // Build full modifier string based on SEL_A, SEL_B, HI_A, HI_B
    // Since we can't concatenate strings in constexpr asm, use separate if branches

    // The 4 sel/phase combinations used in KPAIR:
    // (SEL_A=0, SEL_B=0, HI=0): row 0/2, phase 0, col 0/2
    // (SEL_A=0, SEL_B=1, HI=0): row 0/2, phase 0, col 1/3
    // (SEL_A=1, SEL_B=0, HI=0): row 1/3, phase 0, col 0/2
    // (SEL_A=1, SEL_B=1, HI=0): row 1/3, phase 0, col 1/3
    // Same 4 with HI=1 for phase 1

    #define _DO_MFMA(DP, AP, BP, CP) do { \
        if constexpr (SEL_A==0 && SEL_B==0 && HI_A==0 && HI_B==0) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "", "op_sel_hi:[0,0,0]"); \
        } else if constexpr (SEL_A==0 && SEL_B==1 && HI_A==0 && HI_B==0) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "op_sel:[0,1,0]", "op_sel_hi:[0,0,0]"); \
        } else if constexpr (SEL_A==1 && SEL_B==0 && HI_A==0 && HI_B==0) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "op_sel:[1,0,0]", "op_sel_hi:[0,0,0]"); \
        } else if constexpr (SEL_A==1 && SEL_B==1 && HI_A==0 && HI_B==0) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "op_sel:[1,1,0]", "op_sel_hi:[0,0,0]"); \
        } else if constexpr (SEL_A==0 && SEL_B==0 && HI_A==1 && HI_B==1) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "", "op_sel_hi:[1,1,0]"); \
        } else if constexpr (SEL_A==0 && SEL_B==1 && HI_A==1 && HI_B==1) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "op_sel:[0,1,0]", "op_sel_hi:[1,1,0]"); \
        } else if constexpr (SEL_A==1 && SEL_B==0 && HI_A==1 && HI_B==1) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "op_sel:[1,0,0]", "op_sel_hi:[1,1,0]"); \
        } else if constexpr (SEL_A==1 && SEL_B==1 && HI_A==1 && HI_B==1) { \
            _FP4_ASM_BODY(DP, AP, BP, CP, "op_sel:[1,1,0]", "op_sel_hi:[1,1,0]"); \
        } \
    } while(0)

    // Dispatch D/A/B/C VGPR/AGPR prefixes (16 combinations)
    if constexpr (dA && aA && bA && cA) { _DO_MFMA("a","a","a","a"); }
    else if constexpr (dA && aA && bA && !cA) { _DO_MFMA("a","a","a","v"); }
    else if constexpr (dA && aA && !bA && cA) { _DO_MFMA("a","a","v","a"); }
    else if constexpr (dA && !aA && bA && cA) { _DO_MFMA("a","v","a","a"); }
    else if constexpr (!dA && aA && bA && cA) { _DO_MFMA("v","a","a","a"); }
    else if constexpr (dA && aA && !bA && !cA) { _DO_MFMA("a","a","v","v"); }
    else if constexpr (dA && !aA && bA && !cA) { _DO_MFMA("a","v","a","v"); }
    else if constexpr (dA && !aA && !bA && cA) { _DO_MFMA("a","v","v","a"); }
    else if constexpr (!dA && aA && bA && !cA) { _DO_MFMA("v","a","a","v"); }
    else if constexpr (!dA && aA && !bA && cA) { _DO_MFMA("v","a","v","a"); }
    else if constexpr (!dA && !aA && bA && cA) { _DO_MFMA("v","v","a","a"); }
    else if constexpr (dA && !aA && !bA && !cA) { _DO_MFMA("a","v","v","v"); }
    else if constexpr (!dA && aA && !bA && !cA) { _DO_MFMA("v","a","v","v"); }
    else if constexpr (!dA && !aA && bA && !cA) { _DO_MFMA("v","v","a","v"); }
    else if constexpr (!dA && !aA && !bA && cA) { _DO_MFMA("v","v","v","a"); }
    else { _DO_MFMA("v","v","v","v"); }

    #undef _DO_MFMA
    #undef _FP4_ASM_BODY
}
#undef _MFMA_FP4_REG

} // namespace macros
} // namespace kittens