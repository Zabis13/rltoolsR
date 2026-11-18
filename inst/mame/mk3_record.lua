-- Recorder for MK3 (Genesis): logs pad presses of a human player and RAM,
-- to learn the menu path (Start, fighter select, intro) for automated
-- snapshots. Run MAME in a window with -autoboot_script; output in RLT_REC
-- (default /tmp/rlt_mame/record):
--   events.tsv  frame, field, 1 pressed / 0 released, then key RAM values
--   ram.bin     records: frame u32 LE, reason u8 (0 every 15 frames,
--               1 at an input event), 65536 bytes of work RAM 0xFF0000..

local DIR = os.getenv("RLT_REC") or "/tmp/rlt_mame/record"
os.execute("mkdir -p '" .. DIR .. "'")
local machine = manager.machine
local space   = machine.devices[":maincpu"].spaces["program"]
local screen  = machine.screens:at(1)

local fields = {}
for _, port in pairs(machine.ioport.ports) do
  for name, field in pairs(port.fields) do
    if name:match("^P[12] ") then fields[#fields + 1] = { port = port, field = field, name = name, on = false } end
  end
end

local ev  = assert(io.open(DIR .. "/events.tsv", "w"))
local ram = assert(io.open(DIR .. "/ram.bin", "wb"))
local KEYS = { sel1 = 0xFF0127, sel2 = 0xFF0129, id1 = 0xFFCB4B, id2 = 0xFFCB97,
               hp1 = 0xFFF865, hp2 = 0xFFF9D7, t10 = 0xFFF5A3, t1 = 0xFFF5A5 }
local key_names = { "sel1", "sel2", "id1", "id2", "hp1", "hp2", "t10", "t1" }
ev:write("frame\tfield\tstate\t" .. table.concat(key_names, "\t") .. "\n")

local function dump(frame, reason)
  local parts = { string.pack("<I4B", frame & 0xffffffff, reason) }
  for a = 0xFF0000, 0xFFFFFF, 4096 do
    local chunk = {}
    for i = 0, 4095 do chunk[#chunk + 1] = string.char(space:read_u8(a + i)) end
    parts[#parts + 1] = table.concat(chunk)
  end
  ram:write(table.concat(parts))
end

local function on_frame()
  local frame = screen:frame_number()
  local changed = false
  for _, f in ipairs(fields) do
    local v = f.port:read()
    local on = (v & f.field.mask) ~= (f.field.defvalue & f.field.mask)
    if on ~= f.on then
      f.on = on; changed = true
      local vals = {}
      for _, k in ipairs(key_names) do vals[#vals + 1] = tostring(space:read_u8(KEYS[k])) end
      ev:write(string.format("%d\t%s\t%d\t%s\n", frame, f.name, on and 1 or 0, table.concat(vals, "\t")))
    end
  end
  if changed then ev:flush(); dump(frame, 1)
  elseif frame % 15 == 0 then dump(frame, 0) end
end

rlt_rec_sub = emu.add_machine_frame_notifier(function()
  local ok, err = pcall(on_frame)
  if not ok then print("rlt_rec: error: " .. tostring(err)) end
end)
rlt_rec_stop = emu.add_machine_stop_notifier(function() ev:close(); ram:close() end)
print("rlt_rec: recording to " .. DIR)
