"""Routing, dispatch, and combine checked against PyTorch; no expert GEMMs are timed here."""
from common import F, I, P, Operation, bind, checked, library, pointers, stream, torch
from reference.python.moe import route, dispatch, combine


def cases(args, family):
    lib = library("expert_dispatch")
    routing = bind(lib, "dscuda_route", [P] * 6 + [I] * 3 + [F, P])
    packing = bind(lib, "dscuda_dispatch", [P] * 8 + [I] * 4 + [P])
    combining = bind(lib, "dscuda_combine", [P] * 5 + [I] * 3 + [P])
    combine_backward = bind(lib, "dscuda_combine_backward", [P] * 7 + [I] * 3 + [P])
    unroute = bind(lib, "dscuda_unroute", [P] * 3 + [I] * 3 + [P])
    route_backward = bind(lib, "dscuda_route_backward", [P] * 4 + [I] * 3 + [F, P])
    update_bias = bind(lib, "dscuda_routing_bias", [P] * 2 + [I] * 2 + [F, P])
    shapes = ((7, 11, 4, 2), (65, 129, 16, 4)) if args.test else (
        ((512, 512, 8, 2),) if args.suite == "quick" else ((4096, 512, 64, 8), (8192, 1024, 128, 8)))
    for rows, width, experts, topk in shapes:
        size = f"tokens={rows},D={width},E={experts},topk={topk}"
        logits = torch.randn(rows, experts, device="cuda", requires_grad=True)
        bias = torch.randn(experts, device="cuda") * .1
        x = torch.randn(rows, width, device="cuda")
        scores = torch.empty_like(logits)
        ids = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
        weights = torch.empty(rows, topk, device="cuda")
        counts = torch.empty(experts, device="cuda", dtype=torch.int32)
        scale = 1.25
        def pytorch_route():
            chosen, w = route(logits, bias, topk, scale)
            count = torch.zeros(experts, device="cuda", dtype=torch.int32)
            count.scatter_add_(0, chosen.flatten(), torch.ones(rows * topk, device="cuda", dtype=torch.int32))
            return logits.sigmoid(), chosen.int(), w, count
        def custom_route():
            checked(lib, "expert", routing(*pointers((scores, ids, weights, counts, logits, bias)),
                                            rows, experts, topk, scale, stream()))
            return scores, ids, weights, counts
        expected_route = pytorch_route()
        custom_route()
        yield Operation(size, "fp32", "route", {"custom": custom_route, "PyTorch": pytorch_route}, expected_route)
        packed = torch.empty(rows * topk, width, device="cuda")
        offsets = torch.empty(experts + 1, device="cuda", dtype=torch.int32)
        maps = [torch.empty(rows * topk, device="cuda", dtype=torch.int32) for _ in range(3)]
        route_to_slot, slot_to_route, slot_expert = maps
        def custom_dispatch():
            checked(lib, "expert", packing(*pointers((packed, offsets, *maps, x, ids, counts)),
                                            rows, width, experts, topk, stream()))
            return packed, offsets, *maps
        def pytorch_dispatch():
            p, o, slots = dispatch(x, ids.long(), experts)
            slots = slots.flatten()
            reverse = slots.argsort()
            return p, o.int(), slots.int(), reverse.int(), ids.flatten()[reverse]
        def canonical(result):
            p, o, slots, reverse, expert = result
            return p[slots.long()], o, reverse[slots.long()], expert[slots.long()]
        expected_dispatch = canonical(pytorch_dispatch())
        custom_dispatch()
        yield Operation(size, "fp32", "dispatch", {"custom": custom_dispatch, "PyTorch": pytorch_dispatch},
                        expected_dispatch, normalize=canonical)
        # Dispatch slot ordering is intentionally free; subsequent checks use its actual map.
        packed_output = torch.randn_like(packed).requires_grad_()
        shared = torch.randn_like(x).requires_grad_()
        w = weights.detach().clone().requires_grad_()
        slots = route_to_slot.long().reshape(rows, topk)
        output = torch.empty_like(x)
        def pytorch_combine():
            return combine(packed_output, slots, w, shared)
        def custom_combine():
            checked(lib, "expert", combining(*pointers((output, shared, packed_output, w, route_to_slot)),
                                              rows, width, topk, stream()))
            return output
        expected_output = pytorch_combine()
        yield Operation(size, "fp32", "combine", {"custom": custom_combine, "PyTorch": pytorch_combine},
                        (expected_output,))
        if args.test:
            dout = torch.randn_like(x)
            dp, dw, ds = torch.empty_like(packed_output), torch.empty_like(w), torch.empty_like(shared)
            def pytorch_combine_backward():
                return torch.autograd.grad(expected_output, (packed_output, w, shared), dout, retain_graph=True)
            def custom_combine_backward():
                checked(lib, "expert", combine_backward(*pointers((dp, dw, ds, dout, packed_output, w, route_to_slot)),
                                                        rows, width, topk, stream()))
                return dp, dw, ds
            yield Operation(size, "fp32", "combine_backward",
                            {"custom": custom_combine_backward, "PyTorch": pytorch_combine_backward},
                            pytorch_combine_backward())
            dpacked, dx = torch.randn_like(packed), torch.empty_like(x)
            def custom_unroute():
                dx.zero_()
                checked(lib, "expert", unroute(*pointers((dx, dpacked, route_to_slot)), rows, width, topk, stream()))
                return dx
            yield Operation(size, "fp32", "unroute_backward", {"custom": custom_unroute},
                            (dpacked[slots].sum(1),))
            dweights, dlogits = torch.randn_like(weights), torch.empty_like(logits)
            def custom_route_backward():
                checked(lib, "expert", route_backward(*pointers((dlogits, dweights, scores, ids)),
                                                       rows, experts, topk, scale, stream()))
                return dlogits
            expected_dlogits = torch.autograd.grad(expected_route[2], logits, dweights)[0]
            yield Operation(size, "fp32", "route_backward", {"custom": custom_route_backward}, (expected_dlogits,))
            updated = torch.empty_like(bias)
            def custom_update():
                updated.copy_(bias)
                checked(lib, "expert", update_bias(*pointers((updated, counts)), experts, rows * topk, .01, stream()))
                return updated
            expected_bias = bias + .01 * torch.sign(rows * topk / experts - counts.float())
            yield Operation(size, "fp32", "routing_bias", {"custom": custom_update}, (expected_bias,))
