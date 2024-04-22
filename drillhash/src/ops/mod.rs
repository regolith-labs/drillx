mod add;
mod addl;
mod div;
mod mask;
mod mul;
mod mull;
// mod sqrt;
mod sub;
mod subl;
// mod xor;

use add::*;
use addl::*;
use div::*;
use mull::*;
use subl::*;
// use left::*;
// use mask::*;
use mul::*;
// use right::*;
// use sqrt::*;
use sub::*;
// use xor::*;

use enum_dispatch::enum_dispatch;

use crate::read_noise;

// TODO Fix div
// TODO Fix sqrt
// TODO Left rotating ops?
// TODO Float ops?

#[derive(Debug, strum::EnumIter)]
#[enum_dispatch(Op)]
pub enum RandomOp {
    AddR(AddR),
    AddL(AddL),
    SubR(SubR),
    SubL(SubL),
    MulR(Mul),
    MulL(MulL),
    // Div(Div),
    // Sqrt(Sqrt),
}

#[enum_dispatch]
pub trait Op {
    fn op(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]) -> bool;
    fn update_state(&mut self, addr: &mut u64, challenge: [u8; 32], nonce: [u8; 8], noise: &[u8]);
}

pub fn random_op<'a>(
    ops: &'a mut [RandomOp],
    addr: &mut u64,
    challenge: [u8; 32],
    nonce: [u8; 8],
    noise: &[u8],
) -> &'a mut RandomOp {
    let seed = read_noise(addr, challenge, nonce, noise);
    let opcode = usize::from_le_bytes(seed);
    &mut ops[opcode % ops.len()]
}
