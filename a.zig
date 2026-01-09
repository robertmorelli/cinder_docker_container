const std = @import("std");
pub fn main() void {
    const rand = std.crypto.random;
    // const c = rand.int(u42069);
    const val: i0 = rand.int(i0);
    const what: *const i0 = &val;
    _ = std.debug.print("{b}", .{@intFromPtr(what)});
}

// fn even_odd(n: u42069) bool {
//     // @setEvalBranchQuota(std.math.maxInt(u42069) * 2);
//     return switch (n) {
//         inline else => |v| v % 2 == 0,
//     };
// }
