-- Makes a two-player (VS) snapshot of MK3 (Genesis) at the start of round 1.
-- From power-on: menu presses at fixed frames (recorded), then both players
-- move their select cursors to the target cells at the same time and
-- confirm (P1 B, P2 A); then P1 Start every 120 frames until the fight is on
-- (both HP 166, timer 99); save state 20 frames later.
-- 0xFF0127 / 0xFF0129 hold the cursor cell on the select screen (grid
-- 5 x 3: row cell // 5, column cell % 5; cell 7 is skipped) and the fighter
-- id in the fight; cell != id (table in mame_make_snapshot.R).
-- RLT_MODE=1p: one player against the CPU; only P1 selects, then Left a few
-- times on the tower screen (tower 1, the leftmost); the CPU opponent is
-- chosen by the game.
--   RLT_P1, RLT_P2  target cells 0..14 (RLT_P2 not used with 1p)
--   RLT_SAVE        state name: <state_directory>/genesis/<name>.sta
--   RLT_KEEP=1      after saving: throttle and keep running (to watch);
--                   otherwise exit
-- Log line on success: "rlt_snap: saved <name> id1 <a> id2 <b>".

local ONE = os.getenv("RLT_MODE") == "1p"
local P = { tonumber(assert(os.getenv("RLT_P1"))), tonumber(os.getenv("RLT_P2") or "0") }
local SAVE = os.getenv("RLT_SAVE") or "snap"
local KEEP = os.getenv("RLT_KEEP") == "1"
local machine = manager.machine
local space   = machine.devices[":maincpu"].spaces["program"]
local screen  = machine.screens:at(1)
local r8 = function(a) return space:read_u8(a) end

local fields = {}
for _, port in pairs(machine.ioport.ports) do
  for name, field in pairs(port.fields) do fields[name] = field end
end
local function f(name) return assert(fields[name], "no input field " .. name) end

-- Menu presses (frame, field), held 6 frames: from a recorded session
local MENU = ONE and { { 828, "P1 B" }, { 2162, "P1 Start" }, { 2264, "P1 Start" } }
  or { { 334, "P1 B" }, { 817, "P1 A" }, { 1131, "P1 A" }, { 1191, "P1 Start" },
       { 1285, "P1 Start" }, { 1431, "P2 Start" } }
local SEL  = { 0xFF0127, 0xFF0129 }
local CONF = { f("P1 B"), f("P2 A") }
local HOLD, GAP = 4, 8

local pl = {}
for p = 1, 2 do
  pl[p] = { up = f("P" .. p .. " Up"), down = f("P" .. p .. " Down"),
            left = f("P" .. p .. " Left"), right = f("P" .. p .. " Right"),
            pressed = nil, t = 0, done = false }
end
if ONE then pl[2].done = true end

local phase, t_phase, round_frame = "menu", 0, nil

local function step_player(p)
  local s = pl[p]
  if s.done then return end
  s.t = s.t + 1
  if s.pressed then
    if s.t >= HOLD then s.pressed:set_value(0); s.pressed = nil; s.t = 0 end
    return
  end
  if s.t < GAP then return end
  local cur = r8(SEL[p])
  if cur > 14 then return end
  local tr, tc, cr, cc = P[p] // 5, P[p] % 5, cur // 5, cur % 5
  local fld
  if cur == P[p] then fld = CONF[p]; s.done = true
  elseif tr > cr then fld = s.down elseif tr < cr then fld = s.up
  elseif tc > cc then fld = s.right else fld = s.left end
  fld:set_value(1); s.pressed = fld; s.t = 0
  print(string.format("rlt_snap: frame %d P%d cursor %d target %d press %s",
                      screen:frame_number(), p, cur, P[p], fld.name))
  if s.done then s.done = "release" end
end

local function on_frame()
  local frame = screen:frame_number()
  if phase == "menu" then
    for _, m in ipairs(MENU) do
      if frame == m[1] then f(m[2]):set_value(1) end
      if frame == m[1] + 6 then f(m[2]):set_value(0) end
    end
    local s2 = r8(SEL[2])
    local ready = ONE and s2 == 255 or (not ONE and s2 ~= 0 and s2 ~= 255)
    if frame > MENU[#MENU][1] + 6 and ready then
      phase = "select"; t_phase = 0
      print(string.format("rlt_snap: frame %d select, cursors %d %d", frame, r8(SEL[1]), s2))
    end
  elseif phase == "select" then
    for p = 1, 2 do
      if pl[p].done == "release" then
        pl[p].t = pl[p].t + 1
        if pl[p].t >= HOLD then CONF[p]:set_value(0); pl[p].done = true end
      else step_player(p) end
    end
    if pl[1].done == true and pl[2].done == true then
      phase = ONE and "tower" or "intro"; t_phase = 0
    end
  elseif phase == "tower" then
    -- Left at 200, 240, ..., 400 frames after the confirm (tower screen at
    -- ~250 in the recording); extra presses at the left edge do nothing.
    t_phase = t_phase + 1
    local k = t_phase - 200
    if k >= 0 and k % 40 == 0 then f("P1 Left"):set_value(1) end
    if k >= 0 and k % 40 == 6 then f("P1 Left"):set_value(0) end
    if t_phase > 410 then phase = "intro"; t_phase = 0 end
  elseif phase == "intro" then
    t_phase = t_phase + 1
    local fight = r8(0xFFF865) == 166 and r8(0xFFF9D7) == 166 and
                  r8(0xFFF5A3) == 9 and r8(0xFFF5A5) == 9
    if not fight then
      local k = t_phase % 120
      f("P1 Start"):set_value((t_phase > 60 and k < 6) and 1 or 0)
      round_frame = nil
    else
      f("P1 Start"):set_value(0)
      round_frame = round_frame or frame
      if frame - round_frame == 20 then
        machine:save(SAVE)
        print(string.format("rlt_snap: saved %s id1 %d id2 %d (frame %d)", SAVE,
                            r8(0xFFCB4B), r8(0xFFCB97), frame))
        phase = "done"; t_phase = 0
        if KEEP then machine.video.throttled = true end
      end
    end
  elseif phase == "done" then
    t_phase = t_phase + 1
    if not KEEP and t_phase == 10 then machine:exit() end
  end
  if phase ~= "done" and frame > 30000 then
    print("rlt_snap: timeout in phase " .. phase); machine:exit()
  end
end

if KEEP then machine.video.throttled = false end
rlt_snap_sub = emu.add_machine_frame_notifier(function()
  local ok, err = pcall(on_frame)
  if not ok then print("rlt_snap: error: " .. tostring(err)); machine:exit() end
end)
