-- Mortal Kombat 3 (Sega Genesis, MAME driver "genesis", -ctrlN md6button):
-- the MAME side of rl_mame_env("mk3").
--
-- Lockstep over two FIFOs. Every K emulated frames the script sends one
-- observation to R and blocks until R answers with a command.
-- Environment variables (set by R):
--   RLT_ACT, RLT_OBS  FIFO paths (R -> MAME commands, MAME -> R observations)
--   RLT_K             emulated frames per step
--   RLT_START=1       load state "rlt", run RLT_INTRO frames (input lock at
--                     the start of a round), save state "rltgo", then the
--                     first observation
--   RLT_INTRO         intro frames, run once at the start
--   RLT_TOWER         tower mode instead of RLT_START: select cell 0..14 of
--                     P1's fighter (see the tower block below)
--   RLT_WATCHDOG      tower mode: frames without a change of the round
--                     timer before a console reset (default 3600)
--   RLT_RETRY         tower mode: "checkpoint" (save state "rltck" at the
--                     start of every fight, load it after a lost match) or
--                     "continue" (Continue in the game menu)
--   RLT_NTOWERS       tower mode: number of towers; after the last one tower 1--
-- Command, 4 bytes: cmd, P1 action, P2 action, reserved.
--   cmd 0 step, 1 save state "rltgo", 2 load state "rltgo", 3 quit.
-- Action a in 0..62: direction = a // 7, button = a % 7.
--   Direction, relative to the opponent: 0 none, 1 up, 2 up-forward,
--   3 forward, 4 down-forward, 5 down, 6 down-back, 7 back, 8 up-back.
--   Button: 0 none, 1..6 = pad A, B, C, X, Y, Z. B (block) and Y (run) are
--   held all K frames like the direction, the others pressed in the first
--   frame only.
-- Observation, 18 bytes: frame (uint32 little-endian), then big-endian
--   hp1 u8, hp2 u8, x1 u16, x2 u16, y1 i16, y2 i16, timer u8, won1 u8,
--   won2 u8, air u8 (bit 0: P1 in the air, bit 1: P2, bit 2: paused --
--   tower mode, the script plays (Continue, reboot), the action is ignored).

local ACT = assert(os.getenv("RLT_ACT"))
local OBS = assert(os.getenv("RLT_OBS"))
local K   = tonumber(os.getenv("RLT_K") or "4")
local INTRO = math.max(tonumber(os.getenv("RLT_INTRO") or "0"), 1)

-- RAM map (see the README)
local HP1, HP2 = 0xFFF865, 0xFFF9D7
local X1, X2   = 0xFFCB1C, 0xFFCB68
local Y1, Y2   = 0xFFCB20, 0xFFCB6C
local T10, T1  = 0xFFF5A3, 0xFFF5A5
local W1, W2   = 0xFFF86B, 0xFFF9DD
local AIR1, AIR2 = 0xFFCB12, 0xFFCB5E   -- bit 7: in the air
local HOLD = { [2] = true, [5] = true }

local machine = manager.machine
local space   = machine.devices[":maincpu"].spaces["program"]
local screen  = machine.screens:at(1)

local fields = {}
for _, port in pairs(machine.ioport.ports) do
  for name, field in pairs(port.fields) do fields[name] = field end
end
local function f(p, name)
  local field = fields[string.format("P%d %s", p, name)]
  if not field then error(string.format("no input field 'P%d %s'", p, name)) end
  return field
end
local function dirs(U, D, B, F)   -- up, down, back, forward
  return { [1] = { U }, [2] = { U, F }, [3] = { F }, [4] = { D, F },
           [5] = { D }, [6] = { D, B }, [7] = { B }, [8] = { U, B } }
end
-- DIR_R[p]: opponent on the right (forward = Right), DIR_L[p]: on the left.
local DIR_R, DIR_L, BTN = {}, {}, {}
for p = 1, 2 do
  local U, D, L, R = f(p, "Up"), f(p, "Down"), f(p, "Left"), f(p, "Right")
  DIR_R[p], DIR_L[p] = dirs(U, D, L, R), dirs(U, D, R, L)
  BTN[p] = {}
  for b, name in ipairs({ "A", "B", "C", "X", "Y", "Z" }) do BTN[p][b] = { f(p, name) } end
end

local act = assert(io.open(ACT, "rb"))
local obs = assert(io.open(OBS, "wb"))

local held, btn_on = { {}, {} }, { {}, {} }
local opp_left = { false, true }
local count = 0

local function set_fields(list, v)
  for _, field in ipairs(list) do field:set_value(v) end
end

local function apply(p, a)
  set_fields(held[p], 0)
  set_fields(btn_on[p], 0)
  held[p], btn_on[p] = {}, {}
  local x1, x2 = space:read_u16(X1), space:read_u16(X2)
  if x1 ~= x2 then opp_left[1], opp_left[2] = x2 < x1, x1 < x2 end
  local d, b = a // 7, a % 7
  local dir = opp_left[p] and DIR_L[p] or DIR_R[p]
  for _, field in ipairs(dir[d] or {}) do held[p][#held[p] + 1] = field end
  local dst = HOLD[b] and held[p] or btn_on[p]
  for _, field in ipairs(BTN[p][b] or {}) do dst[#dst + 1] = field end
  set_fields(held[p], 1)
  set_fields(btn_on[p], 1)
end

local function send_obs(paused)
  local r8 = function(a) return space:read_u8(a) end
  local r16 = function(a) return space:read_u16(a) end
  local i16 = function(a) return space:read_i16(a) end
  local air = ((r8(AIR1) >> 7) & 1) | (((r8(AIR2) >> 7) & 1) << 1) | (paused and 4 or 0)
  obs:write(string.pack("<I4", screen:frame_number() & 0xffffffff) ..
            string.pack(">BBHHhhBBBB", r8(HP1), r8(HP2), r16(X1), r16(X2),
                        i16(Y1), i16(Y2), 10 * r8(T10) + r8(T1), r8(W1), r8(W2), air))
  obs:flush()
end

-- Tower mode (RLT_TOWER): no snapshots. From power-on, and after every
-- reboot, the script itself (no observations) passes the 1P menu, moves the
-- select cursor to the cell and confirms, picks tower 1 (Left on the tower
-- screen) and presses Start until the fight is on (both HP 166, timer 99);
-- then lockstep, between fights too. When P2 wins its 2nd round (lost
-- match), Start every 120 frames until the next fight is on (Continue).
-- When the round timer has not changed for WATCHDOG frames (end of the
-- tower, game over): soft reset of the console and boot again. Menu press
-- frames are from a recorded session (as in mk3_snapshot.lua).
-- RLT_RETRY=checkpoint: at the start of every fight (round 1, both win
-- counters 0) the state is saved as "rltck"; a lost match loads it instead
-- of Continue. A reboot after a won match (end of the tower) picks the next
-- tower: Left 6 times (tower 1), then Right tower_no - 1 times.
local TOWER = tonumber(os.getenv("RLT_TOWER") or "")
local WATCHDOG = tonumber(os.getenv("RLT_WATCHDOG") or "3600")
local CKPT = os.getenv("RLT_RETRY") == "checkpoint"
local NTOWERS = tonumber(os.getenv("RLT_NTOWERS") or "1")
local tower_no = 1
local have_ck, at_start, won_match, prev_won1 = false, false, false, 0
local SEL1, SEL2 = 0xFF0127, 0xFF0129
local MENU1P = { { 828, "B" }, { 2162, "Start" }, { 2264, "Start" } }
local START = f(1, "Start")
local boot = TOWER and "menu" or nil      -- nil: lockstep
local booted = false                      -- first fight reached: boots go on in lockstep, paused
local boot_t, sel_t, sel_pressed, confirmed = 0, 0, nil, false
local prev_won2, cont_pending = 0, false
local last_timer, timer_still = -1, 0
local CONT_MAX = 6000
local reboot   -- defined below

local function fight_on()
  return space:read_u8(HP1) == 166 and space:read_u8(HP2) == 166 and
         space:read_u8(T10) == 9 and space:read_u8(T1) == 9
end

local function start_pulse(t)   -- 6 of every 120 frames, after the first 60
  START:set_value((t > 60 and t % 120 < 6) and 1 or 0)
end

local function boot_frame()
  boot_t = boot_t + 1
  if boot == "menu" then
    for _, m in ipairs(MENU1P) do
      if boot_t == m[1] then f(1, m[2]):set_value(1) end
      if boot_t == m[1] + 6 then f(1, m[2]):set_value(0) end
    end
    if boot_t > MENU1P[#MENU1P][1] + 6 and space:read_u8(SEL2) == 255 then
      boot, sel_t = "select", 0
    end
  elseif boot == "cont" then
    -- lost match: Start (Continue) until the select screen, then as at boot;
    -- no select screen within CONT_MAX frames (game over): reboot
    if space:read_u8(SEL2) == 255 then
      START:set_value(0)
      boot, sel_t, sel_pressed, confirmed = "select", 0, nil, false
      print(string.format("rlt: continue, select screen after %d frames", boot_t))
    elseif boot_t > CONT_MAX then
      reboot(); return
    else
      start_pulse(boot_t)
    end
  elseif boot == "select" then
    sel_t = sel_t + 1
    if sel_pressed then
      if sel_t >= 4 then
        sel_pressed:set_value(0); sel_pressed = nil; sel_t = 0
        if confirmed then boot = "tower" end
      end
    elseif sel_t >= 8 then
      local cur = space:read_u8(SEL1)
      if cur <= 14 then
        local tr, tc, cr, cc = TOWER // 5, TOWER % 5, cur // 5, cur % 5
        local name
        if cur == TOWER then name = "B"; confirmed = true
        elseif tr > cr then name = "Down" elseif tr < cr then name = "Up"
        elseif tc > cc then name = "Right" else name = "Left" end
        sel_pressed = f(1, name); sel_pressed:set_value(1); sel_t = 0
      end
    end
  elseif boot == "tower" then
    -- Left at 200, 240, ..., 400 frames after the confirm, then Right at
    -- 440, 480, ... (tower_no - 1 times)
    sel_t = sel_t + 1
    local k = sel_t - 200
    local name = k <= 200 and "Left" or "Right"
    if k >= 0 and k % 40 == 0 then f(1, name):set_value(1) end
    if k >= 0 and k % 40 == 6 then f(1, name):set_value(0) end
    if sel_t > 410 + 40 * (tower_no - 1) then
      boot, sel_t = "intro", 0
      print(string.format("rlt: tower %d", tower_no))
    end
  elseif boot == "intro" then
    sel_t = sel_t + 1
    if fight_on() then
      START:set_value(0)
      boot, count, booted = nil, 0, true
      prev_won2, last_timer, timer_still = 0, -1, 0
      prev_won1, at_start, won_match, have_ck = 0, false, false, false
      print(string.format("rlt: tower fight on, boot frames %d", boot_t))
    else
      start_pulse(sel_t)
    end
  end
  if boot and boot_t > 30000 then error("tower boot timeout in phase " .. boot) end
end

reboot = function()
  apply(1, 0); apply(2, 0); START:set_value(0)
  print(string.format("rlt: reboot, timer still for %d frames", timer_still))
  if CKPT and won_match then
    tower_no = tower_no % NTOWERS + 1
    print(string.format("rlt: tower done, next tower %d", tower_no))
  end
  machine:soft_reset()
  boot, boot_t, sel_t, sel_pressed, confirmed = "menu", 0, 0, nil, false
  cont_pending = false
end

-- Save / load: release inputs, wait RELEASE_WAIT frames, then save / load
-- -- the pad keeps the last pressed buttons outside the snapshot.
local START_WAIT, RELEASE_WAIT = 8, 4
local pending_op, pending_name, release_left = nil, nil, 0

local function request(op, name)
  apply(1, 0); apply(2, 0)
  pending_op, pending_name, release_left = op, name, RELEASE_WAIT
end

-- Tower mode, every lockstep frame: checkpoint at the start of a fight; a
-- lost match hands over to the script (phase "cont", no observations until
-- the next fight) or, with a checkpoint, marks it for loading; watchdog.
-- Returns true when lockstep stops.
local function tower_frame()
  local won1, won2 = space:read_u8(W1), space:read_u8(W2)
  if won1 >= 2 and prev_won1 < 2 then won_match = true end
  prev_won1 = won1
  local lost = won2 >= 2 and prev_won2 < 2
  prev_won2 = won2
  if lost then cont_pending = true end   -- after the next observation (R sees the KO)
  if CKPT then
    local start_now = won1 == 0 and won2 == 0 and fight_on()
    local save = start_now and not at_start
    at_start = start_now
    if save then
      won_match, have_ck = false, true
      request("save", "rltck")
      print("rlt: fight start, checkpoint saved")
      return true
    end
  end
  local t = 10 * space:read_u8(T10) + space:read_u8(T1)
  if t ~= last_timer then last_timer, timer_still = t, 0 else timer_still = timer_still + 1 end
  if timer_still > WATCHDOG then reboot(); return true end
  return false
end

-- Start: load, run START_WAIT frames, load again, run INTRO frames, save
-- "rltgo" (every restart loads it: no intro per round).
local pending_start = os.getenv("RLT_START") == "1"
local start_stage = 0

local function on_frame()
  if pending_op then
    release_left = release_left - 1
    if release_left == 0 then
      if pending_op == "save" then machine:save(pending_name) else machine:load(pending_name) end
      if pending_name == "rltck" and pending_op == "load" then
        -- the fight starts again: counters of the loaded state, no new save
        prev_won1, prev_won2, at_start, won_match = 0, 0, true, false
        last_timer, timer_still = -1, 0
      end
      pending_op = nil
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
    elseif start_stage == 1 + START_WAIT + INTRO then
      machine:save("rltgo")
      pending_start = false
    end
    count = 0
    return
  end
  if boot then
    boot_frame()
    -- after the first fight (Continue, reboot): lockstep goes on, paused
    -- observations, actions ignored; the first boot sends nothing
    if boot and booted then
      count = count + 1
      if count < K then return end
      count = 0
      send_obs(true)
      local msg = act:read(4)
      if msg == nil or #msg < 4 then
        print("rlt: exit, command FIFO closed"); machine:exit()
      elseif string.byte(msg, 1) == 3 then
        print("rlt: exit, command 3"); machine:exit()
      end
    end
    return
  end
  if TOWER and tower_frame() then return end
  count = count + 1
  if count == 2 then
    for p = 1, 2 do set_fields(btn_on[p], 0); btn_on[p] = {} end
  end
  if count < K then return end
  count = 0
  send_obs()
  local msg = act:read(4)
  if msg == nil or #msg < 4 then
    print("rlt: exit, command FIFO closed")
    machine:exit(); return
  end
  local cmd, a1, a2 = string.byte(msg, 1, 3)
  if cmd == 0 and cont_pending and have_ck then
    -- lost match: the fight again from its checkpoint
    cont_pending = false
    request("load", "rltck")
    print("rlt: match lost, checkpoint loaded")
  elseif cmd == 0 and cont_pending then
    -- lost match: the script takes over until the next fight
    apply(1, 0); apply(2, 0)
    cont_pending, boot, boot_t = false, "cont", 0
    print("rlt: match lost, continue")
  elseif cmd == 0 then
    apply(1, a1); apply(2, a2)
  elseif cmd == 1 or cmd == 2 then
    request(cmd == 1 and "save" or "load", "rltgo")
  else
    print(string.format("rlt: exit, command %d", cmd))
    machine:exit()
  end
end

local function on_frame_safe()
  local ok, err = pcall(on_frame)
  if not ok then
    print("rlt: error, exit: " .. tostring(err))
    machine:exit()
  end
end

rlt_frame_sub = emu.add_machine_frame_notifier(on_frame_safe)
