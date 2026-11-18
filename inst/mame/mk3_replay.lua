-- Replays pad events recorded by mk3_record.lua (events.tsv) frame by frame
-- from power-on, then presses P1 Start every 120 frames while no fight is on
-- (HP 0 outside a fight) and saves state RLT_SAVE (default "auto") 20 frames
-- after the round starts (both HP 166, timer 99). MAME keeps running.
--   RLT_EVENTS  path to events.tsv
--   RLT_SAVE    state name: <state_directory>/genesis/<name>.sta

local EVENTS = assert(os.getenv("RLT_EVENTS"))
local SAVE   = os.getenv("RLT_SAVE") or "auto"
local machine = manager.machine
local space   = machine.devices[":maincpu"].spaces["program"]
local screen  = machine.screens:at(1)

local fields = {}
for _, port in pairs(machine.ioport.ports) do
  for name, field in pairs(port.fields) do fields[name] = field end
end

local events, last = {}, 0
for line in io.lines(EVENTS) do
  local fr, name, st = line:match("^(%d+)\t([^\t]+)\t(%d)")
  if fr then
    fr = tonumber(fr)
    events[#events + 1] = { frame = fr, field = assert(fields[name], name), on = tonumber(st) }
    if fr > last then last = fr end
  end
end
print(string.format("rlt_replay: %d events, last at frame %d", #events, last))

local r8 = function(a) return space:read_u8(a) end
local start = fields["P1 Start"]
local next_ev, round_frame, saved = 1, nil, false

local function on_frame()
  local frame = screen:frame_number()
  while next_ev <= #events and events[next_ev].frame <= frame do
    local e = events[next_ev]
    e.field:set_value(e.on)
    next_ev = next_ev + 1
  end
  if frame <= last + 30 or saved then return end
  local fight = r8(0xFFF865) == 166 and r8(0xFFF9D7) == 166 and
                r8(0xFFF5A3) == 9 and r8(0xFFF5A5) == 9
  if not fight then
    local k = (frame - last) % 120
    start:set_value((k < 6) and 1 or 0)
    round_frame = nil
    return
  end
  start:set_value(0)
  round_frame = round_frame or frame
  if frame - round_frame == 20 then
    machine:save(SAVE)
    saved = true
    print(string.format("rlt_replay: saved '%s' at frame %d: id1 %d id2 %d sel1 %d sel2 %d",
                        SAVE, frame, r8(0xFFCB4B), r8(0xFFCB97), r8(0xFF0127), r8(0xFF0129)))
  end
end

rlt_replay_sub = emu.add_machine_frame_notifier(function()
  local ok, err = pcall(on_frame)
  if not ok then print("rlt_replay: error: " .. tostring(err)) end
end)
