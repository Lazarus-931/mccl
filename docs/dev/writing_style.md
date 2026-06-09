# mccl C / C++ / Metal style guide


## Language & build
- C++17 for `.cc` / `.h` / `.mm`. Metal Shading Language (MSL) for `.metal`.
- Every unit compiles clean under `clang++ -std=c++17 -Wall -Wextra`. Warnings are defects.

## Naming
- Types: `mccl`-prefixed PascalCase — `mcclSystem`, `mcclTopoGraph`, `mcclResult`.
- Public functions: `mccl`-prefixed lowerCamelCase — `mcclTopoCompute`, `mcclXmlSetAttr`.
- File-internal helpers (in an anonymous namespace): unprefixed lowerCamelCase — `gpuIds`, `bwMatrix`.
- Members and locals: lowerCamelCase — `bw`, `chipCap`, `endpoints`, `nAttrs`.
- `enum class` for domain types, PascalCase enumerators: `enum class mcclNodeType { Gpu, Thunderbolt };`
- Plain `enum` for status/result, `mccl`-prefixed enumerators: `mcclSuccess`, `mcclInvalidArgument`.
- Algorithm/flag enums and macros: `SCREAMING_SNAKE` — `MCCL_ALGO_RING`, `PATH_LOC`, `MAX_STR_LEN`.
- `constexpr` constants: `kMccl…` — `constexpr int kMcclMinChipCap = 30;`

## Comments — terse, no essays (most-enforced mccl rule)
- NO file-header essays, NO SPDX paragraphs, NO `// ==== banner ====` decorations, NO multi-line rationale blocks.
- At most one short line where genuinely non-obvious (a unit, a footgun). Let names carry meaning.
- Close namespaces with a bare `}` — no `}  // namespace mccl`.
- Mechanics: one space after `//` and inside `/* */`; embedded args have none: `f(a, /*force=*/true)`.



## Error handling & API shape
- Functions that can fail return `mcclResult`; outputs come back through `T* out` parameters.
- Validate out-params first: `if (out == nullptr) return mcclInvalidArgument;`
- Return `mcclSuccess` on success. Do not throw across the C-style API.

```cpp
mcclResult mcclTopoGetNode(mcclSystem& sys, int id, mcclNode** out);
```

## File organization
- `.h` headers, `.cc` C++ impl, `.mm` Objective-C++ (Metal/Foundation bridge), `.metal` MSL.
- Headers use `#pragma once` (no include-guard macros).
- Declarations in headers; definitions in `.cc` / `.mm`, not inline in headers (trivial accessors excepted).
- File-internal helpers live in an anonymous `namespace { }`.
- All library code lives in `namespace mccl { … }`.
- Shared definitions (the `mcclResult` enum) live in `src/definitions.h.in`, configured into `definitions.h`.

## Include order
The unit's own header first, then C system headers, then the C++ standard library. Separate the groups
with a blank line; alphabetical within a group. Do not reorder existing groups.

```cpp
#include "topo.h"

#include <sys/socket.h>
#include <unistd.h>

#include <string>
#include <vector>
```

## Hardware & examples (mccl rule)
- Never hardcode real infrastructure: no real IPs, subnets, hostnames, or device ids in source. Route
  them through config (env vars / a cluster file) or runtime discovery — there are no real-cluster defaults.
- Examples use RFC 5737 documentation ranges (`192.0.2.0/24`, `198.51.100.0/24`, `203.0.113.0/24`) and
  generic names (`mac-0`, `mac-1`).

## Metal & Objective-C++
- `.metal` = MSL kernels; `.mm` = Objective-C++ bridging Metal/Foundation to plain C++.
- Keep Objective-C confined to `.mm`. The headers other `.cc` files include expose only plain C++ /
  `mcclResult` — never Obj-C types or Metal objects.
- Wrap command-buffer / autoreleased loops in `@autoreleasepool { }`.
- MSL: explicit address spaces (`device` / `threadgroup` / `constant`); bind buffers with `[[buffer(n)]]`,
  index with `[[thread_position_in_grid]]`; `mccl`-prefix kernel entry points.

---

## Detailed formatting rules

### Line length
100-character limit. Longer lines wrap (see [Line Wrapping](#line-wrapping)).

### Blank lines
- At most **1 blank line** between statements.
- No blank line right after `{` or right before `}`.

### Indentation
2 spaces, no tabs. Continuation lines indent 2 spaces from the statement start, unless aligning after an
open paren is shorter (see [Line Wrapping](#line-wrapping)).

```cpp
mcclResult foo() {
  if (x) {
    bar();
  }
}
```

### Braces
K&R: opening brace on the same line as the control statement or declaration — functions, structs, enums,
namespaces, no exceptions.

```cpp
struct mcclEngine {
  std::string kind;
};

mcclResult bar(int n) {
  for (int i = 0; i < n; ++i) {
    baz(i);
  }
  return mcclSuccess;
}
```

### Spacing
Space between keyword and `(`, space after `)` before `{`, no space inside parens.

```cpp
if (cond) {          // not: if(cond){  /  if ( cond ) {
for (int i = 0; i < n; ++i) {
```

Space after `,` and `;`; none before. `*` and `&` attach to the **type**:

```cpp
mcclResult f(mcclSystem& sys, int* out);
const char* name;
auto* sin = reinterpret_cast<sockaddr_in*>(ifa->ifa_addr);
int *a, *b;          // not allowed — break into separate declarations
```

C-style casts have no trailing space: `(void*)p`, `(uint16_t)pv`.

### Short constructs on a single line
Single-statement `if`/`else` and loops may stay on one line. If the body can't fit, it must use braces;
if either `if` or `else` has braces, both do.

```cpp
if (out == nullptr) return mcclInvalidArgument;
for (const mcclNode& n : sys.nodes) if (n.type == mcclNodeType::Gpu) ++count;
```

### Line wrapping
Bin-pack arguments; wrapped args align under the first argument after `(`.

```cpp
mcclResult mcclTopoAddThunderbolt(mcclSystem& sys, int id, const mcclThunderbolt& tb,
                                  mcclNode** out = nullptr);
```

When even the first argument won't fit, break after `(` and indent 2:

```cpp
mcclResult buildPlan(
  mcclSystem& sys, int algo, mcclTopoGraph* out);
```

Break **after** binary operators, not before:

```cpp
ok = (name.rfind(tok, 0) == 0) ||
     (addr.rfind(tok, 0) == 0);
```

### switch / case
`case` labels at the **same indent** as `switch`.

```cpp
switch (algo) {
case MCCL_ALGO_RING: return computeRing(sys, out);
case MCCL_ALGO_TREE: return computeTree(sys, out);
default:             return mcclInvalidArgument;
}
```

### Enums
One enumerator per line. Two forms:

```cpp
enum class mcclNodeType : int { Gpu = 0, Thunderbolt = 1 };

enum mcclResult {
  mcclSuccess         = 0,
  mcclError           = 1,
  mcclInternalError   = 3,
  mcclInvalidArgument = 4,
};
```

### Initializer lists
No spaces inside `{}` for numeric/uniform init; spaces for string arrays.

```cpp
int endpoints[2] = {-1, -1};
const char* kinds[] = {"gpu", "cpu_amx", "ecore_pool"};
```

### Preprocessor
`#if` / `#else` / `#endif` / `#define` at **column 0** regardless of surrounding indentation. Consecutive
`#define`s are not column-aligned (avoids reflow churn).

```cpp
#define MAX_STR_LEN 255
#define MAX_ATTRS 16
```

### Namespaces
Namespace bodies are **not** indented, and close with a **bare** `}` (no `// namespace` comment).

```cpp
namespace mccl {

mcclResult foo();

}
```

### Templates
`template` on its own line, space before `<`:

```cpp
template <typename T>
T* allocAligned(size_t n);
```
