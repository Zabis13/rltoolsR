-- MAME side of the FIFO lockstep probe (see mame_fifo_probe.R).
--
-- Started by MAME with -autoboot_script. Every K emulated frames the script
-- sends one observation to R and blocks until R answers with a command.
-- Settings come from environment variables set by the R script:
--   RLT_ACT   FIFO path, R -> MAME (commands, 4 bytes each)
--   RLT_OBS   FIFO path, MAME -> R (observations, 4 + RLT_RAM_N bytes)
--   RLT_K     frameskip, emulated frames per step
--   RLT_RAM_ADDR, RLT_RAM_N   RAM region sent with each observation
--   RLT_PORTS file to write the input port/field names to (for mapping)
--
-- Command (4 bytes): cmd, p1 action, p2 action, reserved.
--   cmd 0 step, 1 save state "rlt", 2 load state "rlt", 3 quit.
-- Action a in 0..62: direction = a // 7 (0 none, 1..8), button = a % 7
-- (0 none, 1..6). Direction is held all K frames, button only in the first
-- (except buttons listed in RLT_HOLD, held all K frames). With RLT_RELATIVE
-- directions are relative to the opponent (see below).
-- Observation: frame number (uint32, little-endian), then RAM_N raw bytes.
--
-- Game: Mortal Kombat 3, Sega Genesis cartridge (MAME driver "genesis").
-- 68000, byte-addressed; work RAM at 0xFF0000-0xFFFFFF.
-- Draft: the RAM address is the start of work RAM, not a known variable.

local ACT   = assert(os.getenv("RLT_ACT"))
local OBS   = assert(os.getenv("RLT_OBS"))
local K     = tonumber(os.getenv("RLT_K") or "4")
local ADDR  = tonumber(os.getenv("RLT_RAM_ADDR") or "16711680")  -- 0xFF0000
local RAM_N = tonumber(os.getenv("RLT_RAM_N") or "16")
local PORTS = os.getenv("RLT_PORTS")
-- RLT_HOLD: buttons (1..6, comma-separated) held all K frames like the
-- direction instead of pressed in the first frame only. MK3 block must be
-- held: put the block button here.
local HOLD = {}
for b in string.gmatch(os.getenv("RLT_HOLD") or "", "%d+") do HOLD[tonumber(b)] = true end
-- RLT_DEBUG=1: log the first 200 applied actions (held / one-frame fields).
local DEBUG = os.getenv("RLT_DEBUG") == "1"
local debug_left = 200
print("rlt: RLT_HOLD=" .. tostring(os.getenv("RLT_HOLD")))
-- RLT_START=1: load state "rlt" (copied there by R) before the first
-- observation; the load happens at the end of the first frame.
-- The first load after power-on does not give the same run as later loads
-- (some state outside the snapshot settles in the first frames), so the
-- snapshot is loaded, START_WAIT frames run, and it is loaded again.
local pending_start = os.getenv("RLT_START") == "1"
local START_WAIT = 8
local start_stage = 0
-- RLT_RELATIVE=1 with RLT_X1, RLT_X2 (u16 X addresses of P1, P2): directions
-- are relative to the opponent, "right" (3) = forward = towards him,
-- "left" (7) = back. Side is taken from X at each step; on a tie the last
-- side is kept.
local RELATIVE = os.getenv("RLT_RELATIVE") == "1"
local X1 = tonumber(os.getenv("RLT_X1") or "0")
local X2 = tonumber(os.getenv("RLT_X2") or "0")
local opp_left = { false, false }   -- per player: is the opponent on the left

local machine = manager.machine
local space   = machine.devices[":maincpu"].spaces["program"]
local screen  = machine.screens:at(1)

-- Dumped before the mapping below, so it is there even if a name is missing.
if PORTS then
  local out = assert(io.open(PORTS, "w"))
  for tag, port in pairs(machine.ioport.ports) do
    for name, _ in pairs(port.fields) do out:write(tag, "\t", name, "\n") end
  end
  out:close()
end

-- Input mapping: DIR[p][d] and BTN[p][b] are lists of ioport fields.
-- Six-button pad (MAME started with -ctrlN md6button). Direction d: 0 none,
-- 1 up, 2 up-right, 3 right, 4 down-right, 5 down, 6 down-left, 7 left,
-- 8 up-left. Button b: 0 none, 1..6 = pad A, B, C, X, Y, Z. Which pad button
-- is which attack is set in the game options; the agent does not need it.
local fields = {}
for _, port in pairs(machine.ioport.ports) do
  for name, field in pairs(port.fields) do fields[name] = field end
end
-- One return value: `return assert(x, msg)` would also return msg, and
-- `{ f(...) }` would then hold the message string as a second "field".
local function f(p, name)
  local field = fields[string.format("P%d %s", p, name)]
  if not field then error(string.format("no input field 'P%d %s' (see RLT_PORTS dump)", p, name)) end
  return field
end
-- DIRM[p]: the same with left and right swapped, used in relative mode when
-- the opponent is on the left.
local DIR, DIRM, BTN = {}, {}, {}
local function dirs(U, D, L, R)
  return { [1] = { U }, [2] = { U, R }, [3] = { R }, [4] = { D, R },
           [5] = { D }, [6] = { D, L }, [7] = { L }, [8] = { U, L } }
end
for p = 1, 2 do
  local U, D, L, R = f(p, "Up"), f(p, "Down"), f(p, "Left"), f(p, "Right")
  DIR[p], DIRM[p] = dirs(U, D, L, R), dirs(U, D, R, L)
  BTN[p] = {}
  for b, name in ipairs({ "A", "B", "C", "X", "Y", "Z" }) do BTN[p][b] = { f(p, name) } end
end

-- Same open order as in R (act, then obs), otherwise both sides block.
local act = assert(io.open(ACT, "rb"))
local obs = assert(io.open(OBS, "wb"))

local held   = { {}, {} }   -- fields currently pressed, per player
local btn_on = { {}, {} }   -- button fields to release after the first frame
local count  = 0            -- frames since the last observation

local function set_fields(list, v)
  for _, field in ipairs(list) do field:set_value(v) end
end

local function apply(p, a)
  set_fields(held[p], 0)
  set_fields(btn_on[p], 0)   -- K = 1: never reaches the release at count 2
  local d, b = a // 7, a % 7
  held[p], btn_on[p] = {}, {}
  local dir = DIR[p]
  if RELATIVE then
    local x1, x2 = space:read_u16(X1), space:read_u16(X2)
    if x1 ~= x2 then
      opp_left[1] = x2 < x1
      opp_left[2] = x1 < x2
    end
    if opp_left[p] then dir = DIRM[p] end
  end
  for _, field in ipairs(dir[d] or {}) do held[p][#held[p] + 1] = field end
  local dst = HOLD[b] and held[p] or btn_on[p]
  for _, field in ipairs(BTN[p][b] or {}) do dst[#dst + 1] = field end
  set_fields(held[p], 1)
  set_fields(btn_on[p], 1)
  if DEBUG and p == 1 and a ~= 0 and debug_left > 0 then
    debug_left = debug_left - 1
    local names = function(l) local t = {} for _, x in ipairs(l) do t[#t + 1] = x.name end return table.concat(t, ",") end
    print(string.format("rlt: frame %d P%d a=%d held=[%s] press=[%s]",
                        screen:frame_number(), p, a, names(held[p]), names(btn_on[p])))
  end
end

local function send_obs()
  local parts = { string.pack("<I4", screen:frame_number() & 0xffffffff) }
  for i = 0, RAM_N - 1 do
    parts[#parts + 1] = string.char(space:read_u8(ADDR + i))
  end
  obs:write(table.concat(parts))
  obs:flush()
end

-- Save / load wait RELEASE_WAIT frames with all inputs released first. The
-- pad keeps the last pressed buttons outside the snapshot; loading in the
-- same frame as the release let the game see them right after the load,
-- and the episode then depended on how the previous one ended.
local RELEASE_WAIT = 4
local pending_cmd, release_left = nil, 0

local function on_frame()
  if pending_cmd then
    release_left = release_left - 1
    if release_left == 0 then
      if pending_cmd == 1 then machine:save("rlt") else machine:load("rlt") end
      pending_cmd = nil
    end
    count = 0
    return
  end
  if pending_start then
    start_stage = start_stage + 1
    if start_stage == 1 then
      machine:load("rlt")
    elseif start_stage == 1 + START_WAIT then
      machine:load("rlt")
      pending_start = false
    end
    count = 0
    return
  end
  count = count + 1
  if count == 2 then
    for p = 1, 2 do set_fields(btn_on[p], 0); btn_on[p] = {} end
  end
  if count < K then return end
  count = 0
  send_obs()
  local msg = act:read(4)
  if msg == nil or #msg < 4 then
    print(string.format("rlt: exit, command FIFO closed (read %s bytes)",
                        msg and #msg or "no"))
    machine:exit(); return
  end
  local cmd, a1, a2 = string.byte(msg, 1, 3)
  if cmd == 0 then
    apply(1, a1); apply(2, a2)
  elseif cmd == 1 or cmd == 2 then
    -- Release inputs now, save / load after RELEASE_WAIT frames (see above).
    -- The K frames after a save and after a load then run from the same
    -- state with the same (empty) input and must match.
    apply(1, 0); apply(2, 0)
    pending_cmd, release_left = cmd, RELEASE_WAIT
  else
    print(string.format("rlt: exit, command %d", cmd))
    machine:exit()
  end
end

-- MAME only logs callback errors and keeps running, so R would wait on a
-- half-done step. Stop instead: R then fails on the read timeout.
local function on_frame_safe()
  local ok, err = pcall(on_frame)
  if not ok then
    print("rlt: error, exit: " .. tostring(err))
    machine:exit()
  end
end

-- Keep the subscription referenced, or it is collected and stops firing.
rlt_frame_sub = emu.add_machine_frame_notifier(on_frame_safe)
