import os
import pygame
import asyncio
import threading
from bleak import BleakClient, BleakScanner

from scripts.utils import load_images, Animation
from scripts.entities import Player
from scripts.tilemap import Tilemap
from menu import MainMenu, ConfigMenu, LevelsMenu

class Game:
    def __init__(self):
        pygame.init()
        self.running, self.playing = True, False
        pygame.display.set_caption("juego v1_v12")

        self.base_size = (480, 270)
        self.display = pygame.Surface(self.base_size)
        
        self.window_size = (1280, 720)
        self.fullscreen = False
        self.screen = pygame.display.set_mode(self.window_size)
        
        # ===== FONDO DEL MENÚ =====
        menu_bg_path = os.path.join("data", "images", "background", "menu_b", "1.png")
        self.menu_bg = pygame.image.load(menu_bg_path).convert_alpha()
        self.menu_bg = pygame.transform.scale(self.menu_bg, self.base_size)

        self.blit_size = self.base_size
        self.blit_offset = (0, 0)
        self._recalc_scaling()

        self.clock = pygame.time.Clock()

        # === ASUNTOS DEL MENU ===
        self.main_menu = MainMenu(self)
        self.config_menu = ConfigMenu(self)
        self.levels_menu = LevelsMenu(self)

        self.curr_menu = self.main_menu

        self.selected_level = 1  

        self.DOWN_KEY = False
        self.UP_KEY = False
        self.START_KEY = False
        self.BACK_KEY  = False
        
        # ===== INPUT DE TEXTO (MENÚ CONFIG) =====
        self.TEXT_INPUT = ""
        self.BACKSPACE = False
        
        # ===== PARALLAX BACKGROUND CONSTANTES =====
        self.bg_speeds = [0.25, 0.50, 0.85]
        self.bg_base_speed = 20.0
        self.bg_speed_current = 0.0
        self.bg_accel = 14.0
        self.bg_decel = 14.0
        self.prev_scroll_x = 0.0
        
        # ===== PLAYER ASSETS =====
        self.assets = {
            "ice": load_images("tiles/ice"),
            "water": load_images("tiles/water"),
            "hand": load_images("entities/hand"),
            "heart": load_images("entities/heart"),
            "clock": load_images("entities/clock"),
            "solidwater": load_images("tiles/solidwater"),
            "balloon": load_images("entities/balloon/states"),
            "balloon/pop": load_images("entities/balloon/pop"),
            "player/run": Animation(load_images("entities/player/walk"), img_dur=5),
            "player/idle": Animation(load_images("entities/player/idle"), img_dur=16),
            "player/jump": Animation(load_images("entities/player/dive"), img_dur=5),
            "player/frozen": Animation(load_images("entities/player/frozen"), img_dur=16),
            "player/damage": Animation(load_images("entities/player/damage"), img_dur=5),
        }
        
        # ===== APLICAR TEMA 1 =====
        self.apply_level_theme(1)
        
        # ===== REESCALAR CORAZONES =====
        self.heart_small = []
        for img in self.assets["heart"]:
            w, h = img.get_size()
            scaled_img = pygame.transform.scale(img, (w // 2, h // 2))
            self.heart_small.append(scaled_img)

        # ===== MUNDO =====
        self.tilemap = Tilemap(self, tile_size=16)
        self.tilemap.load("map.json")

        self.player = Player(self, (0, 170), (32, 48))
        self.scroll = [0, 0]
        self.fixed_scroll_y = None
        self.tile_size = 16

        # ===== VARIABLES MODIFICABLES (AUTORUN / TIEMPO PARA PULSAR) =====
        self.auto_speed = 2
        self.fail_return_speed = 1.2
        self.time_limit = 20
        
        self.level_time_limits = {
            1: 10,  # Nivel 1
            2: 5,  # Nivel 2
        }
        
        # ===== VIDAS =====
        self.max_lives = 3
        self.lives = 3
        
        # ===== GAMER TAG =====
        self.gamer_tag = ""
        
        # ===== CONFIGURACIÓN DE NIVEL (5 CHALLENGES) =====
        self.first_stop = 38
        self.challenge_spacing = 60
        self.total_challenges = 5
        
        self.stop_tiles = [self.first_stop + i * self.challenge_spacing for i in range(self.total_challenges)]
        self.stop_index = 0
        self.stop_x_px = self.stop_tiles[self.stop_index] * self.tile_size
        
        self.challenge_stop_indices = set(range(self.total_challenges))
        self.active_challenge = True
        
        # # ===== POSICIÓN DEL GLOBO EN EL MUNDO =====
        self.balloon_tile_x = 45
        self.balloon_world_x = self.balloon_tile_x * self.tilemap.tile_size
        self.balloon_world_y = 32   
        self.balloon_visible = True    
        
        self.balloon_is_popping = False
        self.balloon_pop_frame = 0
        self.balloon_pop_timer = 0.0
        self.balloon_pop_frame_duration = 0.1
        
        # # ===== POSICIÓN DEL BLOQUE FLOTANTE EN EL MUNDO =====
        self.bridge_w_tiles = 10
        self.bridge_h_tiles = 3
        
        self.bridge_tile_x0 = int(self.balloon_tile_x) - (self.bridge_w_tiles // 2)
        self.bridge_world_x = self.bridge_tile_x0 * self.tilemap.tile_size
        self.bridge_world_y = self.balloon_world_y + 32
        
        self.bridge_visible = True
        self.bridge_state = "FLOATING" 
        self.bridge_vel_y = 0.0
        self.bridge_gravity = 400.0
        self.bridge_target_top_tile_y = None
        self.bridge_done = False
        
        self.BRIDGE_VARIANTS = [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 2],
            [7, 8, 8, 8, 8, 8, 8, 8, 8, 3],
            [6, 5, 5, 5, 5, 5, 5, 5, 5, 4],
        ]
        
        # ===== CHALLENGES =====
        self.hand_offset_tiles_y = 8.5
        self.challenges = []
        
        for stop_tile in self.stop_tiles:
            balloon_tile_x = stop_tile + 7
            bridge_tile_x0 = balloon_tile_x - (self.bridge_w_tiles // 2)
        
            self.challenges.append({
                "balloon_tile_x": balloon_tile_x,
                "balloon_world_y": 32,
                "hand_world_y": 32 + (self.hand_offset_tiles_y * self.tile_size),
                "balloon_visible": True,
                "balloon_is_popping": False,
                "balloon_pop_frame": 0,
                "balloon_pop_timer": 0.0,
                "press_count": 0,
                "scored": False,
                "bridge_tile_x0": bridge_tile_x0,
                "bridge_world_x": bridge_tile_x0 * self.tile_size,
                "bridge_world_y": 32 + 32,
                "bridge_visible": True,
                "bridge_state": "FLOATING",
                "bridge_vel_y": 0.0,
                "bridge_target_top_tile_y": None,
                "bridge_done": False,
            })

        # ===== ESTADOS DEL JUEGO =====
        RUNNING = 0
        WAITING = 1
        RESOLVED = 2
        FAILED = 3
        GAME_OVER = 5
        WIN = 6
          
        self.RUNNING = RUNNING
        self.WAITING = WAITING
        self.RESOLVED = RESOLVED
        self.FAILED = FAILED
        self.GAME_OVER = GAME_OVER
        self.WIN = WIN
        
        self.state = self.RUNNING
        
        # ===== PULSAR PARA REVENTAR GLOBO =====
        self.max_presses = 3
        self.time_left = self.time_limit
        self.last_time = pygame.time.get_ticks()
        self.anim_last_time = pygame.time.get_ticks()
        
        self.clock_frames = self.assets["clock"]
        self.hand_frames = self.assets["hand"]
        self.hand_frame_idx = 0
        self.hand_timer = 0.0
        self.hand_frame_duration = 0.10
        self.hand_active = False
        
        self.score = 0
        self.max_score_per_challenge = 1000

        # ===== BLE =====
        self.DEVICE_NAME      = "ESP32-DRV2605"
        self.CHAR_WRITE_UUID  = "abcd1234-ab12-cd34-ef56-abcdef123456"
        self.CHAR_NOTIFY_UUID = "abcd1234-ab12-cd34-ef56-abcdef123457"

        self.INTENSIDAD_ARIADNE = 150
        self.DELAY_ARIADNE      = 200

        self.bluetooth_enabled = False   # el menú lo togglea
        self.ble_client  = None          # BleakClient activo
        self.ble_loop    = None          # loop asyncio del hilo BLE
        self.ble_thread  = None
        
        # ===== FADES (HUD / END / MENSAJES) =====
        self.fade_hearts_in_duration = 0.5
        self.fade_hearts_out_duration = 0.25
        
        self.fade_end_title_duration = 1.0   
        self.fade_msg_duration = 0.5
        
        self.hearts_alpha = 0
        self.hearts_fading_in = True
        self.hearts_fading_out = False
        
        self.end_title_alpha = 0
        self.end_title_fade_in = False
        self.prev_show_end = False
        
        self.msg_alpha = 0
        self.msg_state = "HIDDEN"
        self.msg_interval_index = 0
        self.msg_intervals = [(10, 20), (62, 80), (122, 140), (182, 200), (242, 260)]
        
        # ===== RECUPERACIÓN TRAS FALLAR =====
        self.auto_speed_normal = self.auto_speed
        self.fail_return_active = False
        
        self.recovering = False
        self.escape_started = False
        self.escape_end_offset = 11
        self.escape_tile_N = 49
        
        self.escape_jump_tiles = 6
        self.escape_target_y = None
        self.pending_life_loss = False
        self.go_phase = "NONE"    
        self.go_walk_end_x = None
        
        # ===== BITMAP FONT =====
        self.bitmap_charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789$¤+-*=%”’#@&_(),.;:?|{}<>[]§^~© "
        self.bitmap_font = self.load_numbered_bitmap_font("data/fonts", self.bitmap_charset, start_index=1)
        self.bitmap_spacing = 1
        self.bitmap_font_white = self.recolor_bitmap_font(self.bitmap_font, (255, 255, 255))
        
        # ===== VALIDAR GAMER TAG =====
        self.gamer_tag_alpha = 0
        allowed = set(self.bitmap_charset)
        self.gamer_tag = "".join(ch for ch in self.gamer_tag if ch in allowed)
        
        # ===== BITMAP FONT ESCALADO GRANDE MID Y GAMER TAG =====
        self.bitmap_scale_big = 5
        self.bitmap_spacing_big = self.bitmap_spacing * self.bitmap_scale_big
        self.bitmap_scale_mid = 2.3
        self.bitmap_spacing_mid = self.bitmap_spacing * self.bitmap_scale_mid
        self.bitmap_scale_chibi = 0.8
        self.bitmap_spacing_chibi = self.bitmap_spacing * self.bitmap_scale_chibi
        
        self.bitmap_font_big = {}
        self.bitmap_font_mid = {}
        self.bitmap_font_chibi = {}
        self.bitmap_font_big_white = {}
        self.bitmap_font_mid_white = {}
        self.bitmap_font_chibi_white = {}
        for ch, img in self.bitmap_font.items():
            w, h = img.get_size()
            self.bitmap_font_big[ch] = pygame.transform.scale(img, (w * self.bitmap_scale_big, h * self.bitmap_scale_big))
            self.bitmap_font_mid[ch] = pygame.transform.scale(img, (w * self.bitmap_scale_mid, h * self.bitmap_scale_mid))
            self.bitmap_font_chibi[ch] = pygame.transform.scale(img, (w * self.bitmap_scale_chibi, h * self.bitmap_scale_chibi))

        for ch, img in self.bitmap_font_white.items():
            w, h = img.get_size()
            self.bitmap_font_big_white[ch] = pygame.transform.scale(img, (w * self.bitmap_scale_big, h * self.bitmap_scale_big))
            self.bitmap_font_mid_white[ch] = pygame.transform.scale(img, (w * self.bitmap_scale_mid, h * self.bitmap_scale_mid))
            self.bitmap_font_chibi_white[ch] = pygame.transform.scale(img, (w * self.bitmap_scale_chibi, h * self.bitmap_scale_chibi))



    # Llamar justo antes de iniciar niveles para preparar BLE si está habilitado en el menú de configuración. 
    def start_ble_if_needed(self):
        if self.bluetooth_enabled and self.ble_client is None:
            self.ble_thread = threading.Thread(
                target=self._ble_thread_runner, daemon=True
            )
            self.ble_thread.start()
        if not self.bluetooth_enabled:
            self.stop_ble()  # si se deshabilitó, aseguramos detener cualquier conexión activa

    def stop_ble(self):
        """Llamar al salir del nivel o al togglear OFF desde el menú."""
        self.ble_client = None          # la corutina detecta esto y sale

    def send_vibration(self, intensity: int, delay_ms: int):
        """Usar esto desde cualquier parte del juego."""
        if not self.bluetooth_enabled or self.ble_client is None:
            return
        cmd = f"{intensity},{delay_ms}"
        asyncio.run_coroutine_threadsafe(
            self.ble_client.write_gatt_char(self.CHAR_WRITE_UUID, cmd.encode()),
            self.ble_loop
        )

    # ── Internos ────────────────────────────────────────────
    def _ble_thread_runner(self):
        self.ble_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.ble_loop)
        self.ble_loop.run_until_complete(self._ble_task())

    async def _ble_task(self):
        from bleak import BleakClient, BleakScanner
        device = await BleakScanner.find_device_by_name(self.DEVICE_NAME, timeout=10)
        if device is None:
            print("ESP32 no encontrado")
            return
        async with BleakClient(device) as client:
            self.ble_client = client
            await client.start_notify(self.CHAR_NOTIFY_UUID, self._on_notify)
            while self.ble_client is not None:
                await asyncio.sleep(0.1)
            await client.stop_notify(self.CHAR_NOTIFY_UUID)

    def _on_notify(self, sender, data):
        print(f"[ESP32] {data.decode()}")

    def load_numbered_bitmap_font(self, folder, charset, start_index=1):
        font = {}
        for i, ch in enumerate(charset):
            num = start_index + i
            filename = f"{num:02d}.png"
            path = os.path.join(folder, filename)
            try:
                img = pygame.image.load(path).convert_alpha()
            except FileNotFoundError:
                raise FileNotFoundError(f"No se encontró {path}. Revisa que existan 01.png..40.png")
            font[ch] = img
        return font
    
    def _get_font_metrics(self, font):
            if not hasattr(self, '_font_metrics_cache'):
                self._font_metrics_cache = {}
            key = id(font)
            if key in self._font_metrics_cache:
                return self._font_metrics_cache[key]
            metrics = {}
            for ch, img in font.items():
                try:
                    rect = img.get_bounding_rect()
                except Exception:
                    rect = pygame.Rect(0, 0, img.get_width(), img.get_height())
                metrics[ch] = rect
            self._font_metrics_cache[key] = metrics
            return metrics
    
    def measure_bitmap_text(self, text, font, spacing=1):
        w = 0
        if not text:
            return 0, 0
        metrics = self._get_font_metrics(font)
        baseline = 0
    
        for ch in text:
            img = font.get(ch) or font.get(' ')
            if img is None:
                continue
            rect = metrics.get(ch) or metrics.get(' ')
            rect_bottom = rect.bottom if (rect and rect.width and rect.height) else img.get_height()
            baseline = max(baseline, rect_bottom)
    
        for i, ch in enumerate(text):
            img = font.get(ch) or font.get(' ')
            if img is None:
                continue
            w += img.get_width()
            if i != len(text) - 1:
                w += spacing
                
        h = baseline
        return w, h
    
    def draw_bitmap_text(self, surface, text, x, y, font, spacing=1):
        cx = x
        if not text:
            return
    
        metrics = self._get_font_metrics(font)
        baseline = 0
    
        for ch in text:
            img = font.get(ch) or font.get(' ')
            if img is None:
                continue
            rect = metrics.get(ch) or metrics.get(' ')
            rect_bottom = rect.bottom if (rect and rect.width and rect.height) else img.get_height()
            baseline = max(baseline, rect_bottom)
    
        for ch in text:
            img = font.get(ch) or font.get(' ')
            if img is None:
                cx += spacing
                continue
            rect = metrics.get(ch) or metrics.get(' ')
            rect_bottom = rect.bottom if (rect and rect.width and rect.height) else img.get_height()
            y_off = baseline - rect_bottom
            surface.blit(img, (cx, y + y_off))
            cx += img.get_width() + spacing
        
    def recolor_bitmap_font(self, font, color=(255, 255, 255), bg_colorkey=None):
        new_font = {}
        for ch, glyph in font.items():
            g = glyph.convert_alpha()

            if bg_colorkey is not None:
                g.set_colorkey(bg_colorkey)

            mask = pygame.mask.from_surface(g)
            colored = mask.to_surface(setcolor=(*color, 255), unsetcolor=(0, 0, 0, 0))
            new_font[ch] = colored.convert_alpha()

        return new_font
            
    def _approach_alpha(self, alpha, target, duration, dt):
        if duration <= 0:
            return target
        step = 255.0 * (dt / duration)
        if alpha < target:
            return min(target, alpha + step)
        else:
            return max(target, alpha - step)
        
    def _approach_value(self, value, target, rate, dt):
        if rate <= 0:
            return target
        step = rate * dt
        if value < target:
            return min(target, value + step)
        else:
            return max(target, value - step)
    
    def render_bitmap_text_surface(self, text, font, spacing=1):
        w, h = self.measure_bitmap_text(text, font, spacing=spacing)
        surf = pygame.Surface((max(1, w), max(1, h)), pygame.SRCALPHA)
        self.draw_bitmap_text(surf, text, 0, 0, font, spacing=spacing)
        return surf

    def _update_parallax(self, dt, speed_px_s):
        if speed_px_s <= 0:
            return
        base = speed_px_s * dt
        for i, layer in enumerate(self.bg_moving):
            w = layer.get_width()
            self.bg_offsets[i] = (self.bg_offsets[i] + base * self.bg_speeds[i]) % w
    
    def _draw_parallax(self):
        self.display.blit(self.bg_static, (0, 0))
    
        for i, layer in enumerate(self.bg_moving):
            w = layer.get_width()
            x = -self.bg_offsets[i]
            while x < self.base_size[0]:
                self.display.blit(layer, (int(x), 0))
                x += w

    def _recalc_scaling(self):
        sw, sh = self.screen.get_size()
        bw, bh = self.base_size
        scale = min(sw / bw, sh / bh)
        dw, dh = int(bw * scale), int(bh * scale)
        self.blit_size = (dw, dh)
        self.blit_offset = ((sw - dw) // 2, (sh - dh) // 2)

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.window_size)
        self._recalc_scaling()
        
    def apply_level_theme(self, level_id: int):
        self.current_level_theme = level_id
        suffix = "" if level_id == 1 else "2"
    
        self._reload_level_tiles(suffix)
    
        bg_level_folder = "level1" if level_id == 1 else "level2"
        folder = os.path.join("data", "images", "background", bg_level_folder)
    
        self._load_parallax_background(folder)
    
    def _reload_level_tiles(self, suffix: str):
        def choose_folder(base):
            candidate = f"{base}{suffix}"
            path = os.path.join("data", "images", candidate)
            return candidate if os.path.isdir(path) else base
    
        ice_folder = choose_folder("tiles/ice")
        water_folder = choose_folder("tiles/water")
    
        self.assets["ice"] = load_images(ice_folder)
        self.assets["water"] = load_images(water_folder)
    
    def _load_parallax_background(self, folder_path: str):
        self.bg_folder = folder_path
        self.bg_layers = []
        for i in range(1, 5):
            path = os.path.join(self.bg_folder, f"{i:02d}.png")
            img = pygame.image.load(path).convert_alpha()
            img = pygame.transform.scale(img, self.base_size)
            self.bg_layers.append(img)
    
        self.bg_static = self.bg_layers[0]
        self.bg_moving = self.bg_layers[1:]
        self.bg_offsets = [0.0, 0.0, 0.0]
        
    def current_challenge(self):
        if 0 <= self.stop_index < len(self.challenges):
            return self.challenges[self.stop_index]
        return None
    
    def advance_stop(self):
        self.stop_index += 1
        if self.stop_index < len(self.stop_tiles):
            self.stop_x_px = self.stop_tiles[self.stop_index] * self.tile_size
        else:
            self.stop_x_px = float("inf")
            self.active_challenge = False
        
    def _find_water_top_y_for_bridge(self, bridge_tile_x0):
        x0 = bridge_tile_x0
        x1 = bridge_tile_x0 + self.bridge_w_tiles - 1
        water_ys = []
        for loc, tile in self.tilemap.tilemap.items():
            if tile.get("type") != "water":
                continue
            tx, ty = tile["pos"]
            if x0 <= tx <= x1:
                water_ys.append(ty)
        return min(water_ys) if water_ys else None
    
    def _rect_touches_tile_type(self, rect, tile_type):
        ts = self.tilemap.tile_size
        x0 = rect.left // ts
        x1 = (rect.right - 1) // ts
        y0 = rect.top // ts
        y1 = (rect.bottom - 1) // ts
    
        for tx in range(x0, x1 + 1):
            for ty in range(y0, y1 + 1):
                tile = self.tilemap.tilemap.get(f"{tx};{ty}")
                if tile and tile.get("type") == tile_type:
                    return True
        return False
    
    def reset_level(self, level_id: int):
        self.apply_level_theme(level_id)
    
        self.time_limit = self.level_time_limits.get(level_id, 2)
        self.time_left = self.time_limit
    
        self.tilemap = Tilemap(self, tile_size=16)
        self.tilemap.load("map.json")
        self.tile_size = self.tilemap.tile_size
    
        self.player = Player(self, (0, 170), (32, 48))
        self.scroll = [0, 0]
        self.fixed_scroll_y = None
    
        self.auto_speed = 2
        self.auto_speed_normal = self.auto_speed
        self.fail_return_speed = 1.2
        self.fail_return_active = False
    
        now = pygame.time.get_ticks()
        self.last_time = now
        self.anim_last_time = now
    
        self.lives = self.max_lives
        self.score = 0
    
        self.first_stop = 38
        self.challenge_spacing = 60
        self.total_challenges = 5
        self.stop_tiles = [self.first_stop + i * self.challenge_spacing for i in range(self.total_challenges)]
        self.stop_index = 0
        self.stop_x_px = self.stop_tiles[self.stop_index] * self.tile_size
        self.challenge_stop_indices = set(range(self.total_challenges))
        self.active_challenge = True
    
        self.state = self.RUNNING
    
        self.max_presses = 3
        self.hand_frames = self.assets["hand"]
        self.hand_frame_idx = 0
        self.hand_timer = 0.0
        self.hand_active = False
    
        self.hearts_alpha = 0
        self.hearts_fading_in = True
        self.hearts_fading_out = False
        self.end_title_alpha = 0
        self.end_title_fade_in = False
        self.prev_show_end = False
    
        self.msg_alpha = 0
        self.msg_state = "HIDDEN"
        self.msg_interval_index = 0
    
        self.recovering = False
        self.escape_started = False
        self.escape_target_y = None
        self.pending_life_loss = False
        self.go_phase = "NONE"
        self.go_walk_end_x = None
    
        self.bg_offsets = [0.0, 0.0, 0.0]
        self.bg_speed_current = 0.0
    
        self.challenges = []
        for stop_tile in self.stop_tiles:
            balloon_tile_x = stop_tile + 7
            bridge_tile_x0 = balloon_tile_x - (self.bridge_w_tiles // 2)
            self.challenges.append({
                "balloon_tile_x": balloon_tile_x,
                "balloon_world_y": 32,
                "hand_world_y": 32 + (self.hand_offset_tiles_y * self.tile_size),
                "balloon_visible": True,
                "balloon_is_popping": False,
                "balloon_pop_frame": 0,
                "balloon_pop_timer": 0.0,
                "press_count": 0,
                "scored": False,
                "bridge_tile_x0": bridge_tile_x0,
                "bridge_world_x": bridge_tile_x0 * self.tile_size,
                "bridge_world_y": 32 + 32,
                "bridge_visible": True,
                "bridge_state": "FLOATING",
                "bridge_vel_y": 0.0,
                "bridge_target_top_tile_y": None,
                "bridge_done": False,
            })
    
    def run(self):
        self.start_ble_if_needed()
        # === ROUTER DE NIVELES ===
        if self.selected_level == 1:
            self.run_level_1()
        elif self.selected_level == 2:
            self.run_level_2()  
        else:
            self.run_level_1()
            
    def _run_level_shared(self):
        while self.playing and self.running:
            now_anim = pygame.time.get_ticks()
            frame_dt = (now_anim - self.anim_last_time) / 1000.0
            self.anim_last_time = now_anim
    
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.playing = False
                    return
    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F11:
                        self._toggle_fullscreen()
                    if event.key == pygame.K_ESCAPE:
                        self.playing = False
    
                    if event.key == pygame.K_SPACE and self.state == self.WAITING and self.active_challenge:
                        self.send_vibration(self.INTENSIDAD_ARIADNE, self.DELAY_ARIADNE)
                        cur = self.current_challenge()
                        if not cur:
                            continue
                        cur["press_count"] += 1
                        if cur["press_count"] >= self.max_presses:
                            self.state = self.RESOLVED
                            self.hand_active = False
                            if not cur.get("scored", False):
                                elapsed = max(0.0, self.time_limit - self.time_left)
                                gained = int(max(0.0, (self.time_limit - elapsed)) * self.max_score_per_challenge / self.time_limit)
                                gained = max(0, min(gained, self.max_score_per_challenge))
                                self.score += gained
                                cur["scored"] = True
                            cur["balloon_is_popping"] = True
                            cur["bridge_state"] = "FALLING"
                            cur["bridge_vel_y"] = 0.0
                            cur["bridge_target_top_tile_y"] = self._find_water_top_y_for_bridge(cur["bridge_tile_x0"])
                            
            # ===== CAMARA QUE SIGUE =====
            if self.fixed_scroll_y is None:
                self.fixed_scroll_y = self.scroll[1]
                
            anchor_x = 0.23
            anchor_y = 0.71
            
            target_scroll_x = self.player.rect().centerx - self.display.get_width() * anchor_x
            self.scroll[0] += (target_scroll_x - self.scroll[0]) / 30

            if self.fixed_scroll_y is None:
                self.fixed_scroll_y = self.player.rect().centery - self.display.get_height() * anchor_y
            self.scroll[1] = self.fixed_scroll_y
            
            # ===== PARALLAX SINCRONIZADO A LA CAMARA =====
            self.prev_scroll_x = self.scroll[0]
            
            show_end = (self.state == self.WIN) or (self.state == self.GAME_OVER and self.go_phase == "STOP")
            parallax_active = (not show_end) and (
                (self.state in (self.RUNNING, self.FAILED)) or
                (self.state == self.GAME_OVER and self.go_phase != "STOP")
            )
            
            target_speed = self.bg_base_speed if parallax_active else 0.0
            
            if target_speed > self.bg_speed_current:
                self.bg_speed_current = self._approach_value(self.bg_speed_current, target_speed, self.bg_accel, frame_dt)
            else:
                self.bg_speed_current = self._approach_value(self.bg_speed_current, target_speed, self.bg_decel, frame_dt)
            
            self._update_parallax(frame_dt, self.bg_speed_current)
            self._draw_parallax()
            
            render_scroll = (int(self.scroll[0]), int(self.scroll[1]))
            
            # ===== RENDER PLAYER Y TILEMAP =====
            self.player.render(self.display, offset=render_scroll)
            self.tilemap.render(self.display, offset=render_scroll)
            
            # ===== RENDER BLOQUES (TODOS) =====
            ts = self.tilemap.tile_size
            for ch in self.challenges:
                if ch["bridge_visible"] and ch["bridge_state"] in ("FLOATING", "FALLING", "STUCK"):
                    for ix in range(self.bridge_w_tiles):
                        for iy in range(self.bridge_h_tiles):
                            wx = ch["bridge_world_x"] + ix * ts
                            wy = ch["bridge_world_y"] + iy * ts
                            variant = self.BRIDGE_VARIANTS[iy][ix]
                            ice_img = self.assets["ice"][variant]
                            self.display.blit(ice_img, (wx - render_scroll[0], wy - render_scroll[1]))
                            
            # ===== ANIMACION DE ザ・ハンド =====
            if self.state == self.WAITING and self.hand_active and self.active_challenge:
                self.hand_timer += frame_dt
                while self.hand_timer >= self.hand_frame_duration:
                    self.hand_timer -= self.hand_frame_duration
                    self.hand_frame_idx = (self.hand_frame_idx + 1) % len(self.hand_frames)
            
                cur = self.current_challenge()
                if cur:
                    hand_img = self.hand_frames[self.hand_frame_idx]
                    ts = self.tilemap.tile_size
        
                    hand_world_x = cur["balloon_tile_x"] * ts
                    hand_world_y = cur["hand_world_y"]
                    hand_screen_x = int(hand_world_x - render_scroll[0] - hand_img.get_width() // 2)
                    hand_screen_y = int(hand_world_y - render_scroll[1] - hand_img.get_height() // 2)
            
                    self.display.blit(hand_img, (hand_screen_x, hand_screen_y))
            
            # ===== POSICIÓN DEL GLOBO =====
            balloon_world_x = self.balloon_tile_x * self.tilemap.tile_size
            balloon_world_y = self.balloon_world_y
            balloon_screen_x = balloon_world_x - render_scroll[0]
            balloon_screen_y = balloon_world_y - render_scroll[1]

            # ===== MOVIMIENTO Y ESTADOS DEL PLAYER =====
            move_x = self.auto_speed

            if self.state == self.RUNNING:
                player_right_now = self.player.pos[0] + self.player.size[0]
                player_right_next = player_right_now + move_x
            
                if player_right_now < self.stop_x_px <= player_right_next:
                    move_x = self.stop_x_px - player_right_now
                    if move_x < 0:
                        move_x = 0
            
                    if self.stop_index in self.challenge_stop_indices:
                        cur = self.current_challenge()
                        if cur:
                            cur["press_count"] = 0
                    
                            # ===== ACTIVAR ザ・ハンド EN WAITING =====
                            self.hand_active = True
                            self.hand_timer = 0.0
                            self.hand_frame_idx = 0
                    
                        self.state = self.WAITING
                        self.time_left = self.time_limit
                        self.last_time = pygame.time.get_ticks()
            
            # ===== CONDICIONES DE ESTADOS =====    
            if self.state == self.WAITING:
                move_x = 0
            
            if self.state == self.RESOLVED:
                cur = self.current_challenge()
                move_x = self.auto_speed if (cur and cur["bridge_done"]) else 0
            
            if self.state == self.FAILED:
                move_x = self.auto_speed
                
            if self.state == self.GAME_OVER:
                if self.go_phase == "STOP":
                    move_x = 0
                else:
                    move_x = self.auto_speed
            
            if self.state == self.WIN:
                move_x = 0
                
            # ===== METRO SALTO DEL AGUA =====
            if (self.state in (self.FAILED, self.GAME_OVER)) and self.recovering and not self.escape_started:
                top_right_x = self.player.pos[0] + self.player.size[0]
                if top_right_x >= self.escape_x_px:
                    self.escape_started = True
                    self.escape_target_y = self.player.pos[1] - (self.escape_jump_tiles * self.tile_size)
                    
                    if self.state == self.GAME_OVER:
                        self.go_phase = "RISE"
                        self.player.velocity[1] = -5
                    else:
                        self.player.velocity[1] = -5

            self.player.update(self.tilemap, (move_x, 0))
            
            # ===== BLOQUE OTAKU (TOCA AGUA LE HACE DAÑO) =====
            if self.pending_life_loss:
                if self._rect_touches_tile_type(self.player.rect(), "water"):
                    self.pending_life_loss = False
                    self.lives = max(0, self.lives - 1)
            
                    if self.lives == 0:
                        self.state = self.GAME_OVER
                        self.go_phase = "NONE"
            
            # ===== ASEGURAR SALTO =====
            if (self.state in (self.FAILED, self.GAME_OVER)) and self.recovering and self.escape_started and (self.escape_target_y is not None):
                if self.player.pos[1] <= self.escape_target_y:
                    self.player.pos[1] = self.escape_target_y
                    self.player.velocity[1] = 0
            
                    self.recovering = False
                    self.escape_started = False
                    self.escape_target_y = None
            
                    if self.state == self.GAME_OVER:
                        self.go_phase = "WALK2"
                        self.go_walk_end_x = self.player.pos[0] + (8 * self.tile_size)
                    else:
                        self.state = self.RUNNING
                        
                    if self.fail_return_active:
                        self.auto_speed = self.auto_speed_normal
                        self.fail_return_active = False
                        
            # ===== SECUENCIA GAME OVER =====
            if self.state == self.GAME_OVER and self.go_phase == "WALK2":
                if self.player.pos[0] >= self.go_walk_end_x:
                    self.go_phase = "STOP"
                    self.auto_speed = 0
            
            # ===== ESPERANDING TECLA ESPACIO =====
            if self.state == self.WAITING:
                now = pygame.time.get_ticks()
                dt = (now - self.last_time) / 1000.0
                self.last_time = now
            
                self.time_left -= dt
            
                if self.time_left <= 0:
                    
                    if not self.fail_return_active:
                        self.fail_return_active = True
                        self.auto_speed = self.fail_return_speed
                        
                    self.state = self.FAILED
                    self.hand_active = False
                    failed_stop_tile = self.stop_tiles[self.stop_index]
                    self.escape_tile_N = failed_stop_tile + self.escape_end_offset
                    self.escape_x_px = self.escape_tile_N * self.tile_size
                
                    cur = self.current_challenge()
                    if cur and cur["bridge_state"] == "FLOATING":
                        cur["bridge_state"] = "STUCK"
                
                    if not self.recovering:
                        self.pending_life_loss = True
                        self.recovering = True
                        self.escape_started = False
                        self.escape_target_y = None

                    self.advance_stop()
                    
            # ===== UPDATE DE LA ANIMACION POP (TODOS LOS CHALLENGES) =====
            for ch in self.challenges:
                if ch["balloon_is_popping"] and ch["balloon_visible"]:
                    ch["balloon_pop_timer"] += frame_dt
                    while ch["balloon_pop_timer"] >= self.balloon_pop_frame_duration:
                        ch["balloon_pop_timer"] -= self.balloon_pop_frame_duration
                        ch["balloon_pop_frame"] += 1
                        if ch["balloon_pop_frame"] >= len(self.assets["balloon/pop"]):
                            ch["balloon_is_popping"] = False
                            ch["balloon_visible"] = False
                            break
                    
            # ===== BLOQUE DE HIELO CAYENDO (TODOS LOS CHALLENGES) =====
            for ch in self.challenges:
                if ch["bridge_visible"] and ch["bridge_state"] == "FALLING":
            
                    if ch["bridge_target_top_tile_y"] is None:
                        ch["bridge_state"] = "STUCK"
                        continue
            
                    ch["bridge_vel_y"] += self.bridge_gravity * frame_dt
                    ch["bridge_world_y"] += ch["bridge_vel_y"] * frame_dt
            
                    target_top_y_px = ch["bridge_target_top_tile_y"] * self.tilemap.tile_size
                    if ch["bridge_world_y"] >= target_top_y_px:
                        ch["bridge_world_y"] = target_top_y_px
                        ch["bridge_state"] = "LANDED"
                        ch["bridge_done"] = True

                        x0 = ch["bridge_tile_x0"]
                        y0 = ch["bridge_target_top_tile_y"]
                        for dy in range(self.bridge_h_tiles):
                            for dx in range(self.bridge_w_tiles):
                                x = x0 + dx
                                y = y0 + dy
                                loc = f"{x};{y}"
                                self.tilemap.tilemap[loc] = {
                                    "type": "ice",
                                    "variant": self.BRIDGE_VARIANTS[dy][dx],
                                    "pos": [x, y]}
            
                        ch["bridge_visible"] = False
            
                        if ch is self.current_challenge():
                            self.state = self.RUNNING
                            self.advance_stop()
                            self.hand_active = False
                    
            # ===== RENDER GLOBOS (TODOS) =====
            for ch in self.challenges:
                if not ch["balloon_visible"]:
                    continue
            
                balloon_world_x = ch["balloon_tile_x"] * self.tilemap.tile_size
                balloon_world_y = ch["balloon_world_y"]
                balloon_screen_x = balloon_world_x - render_scroll[0]
                balloon_screen_y = balloon_world_y - render_scroll[1]
            
                if ch["balloon_is_popping"]:
                    if ch["balloon_pop_frame"] < len(self.assets["balloon/pop"]):
                        img = self.assets["balloon/pop"][ch["balloon_pop_frame"]]
                    else:
                        img = None
                else:
                    idx = min(ch["press_count"], 2)
                    img = self.assets["balloon"][idx]
            
                if img:
                    self.display.blit(
                        img,
                        (int(balloon_screen_x - img.get_width() // 2),
                         int(balloon_screen_y - img.get_height() // 2)))
                    
            # ===== WIN =====
            if self.state not in (self.GAME_OVER, self.WIN):
                if self.stop_index >= self.total_challenges and self.lives > 0:
                    if self.player.pos[0] + self.player.size[0] >= 308 * self.tile_size:
                        self.state = self.WIN
                        self.auto_speed = 0
                        
            # ===== SHOW END =====
            show_end = (self.state == self.WIN) or (self.state == self.GAME_OVER and self.go_phase == "STOP")
            
            # ===== DETECCION PARA FADES =====
            if show_end and not self.prev_show_end:
                self.hearts_fading_out = True
                self.hearts_fading_in = False
            
                self.end_title_alpha = 0
                self.end_title_fade_in = True
            
            self.prev_show_end = show_end
            
            # ===== ACTUALIZAR FADE CORAZONES =====
            if self.hearts_fading_in:
                self.hearts_alpha = self._approach_alpha(self.hearts_alpha, 255, self.fade_hearts_in_duration, frame_dt)
                if self.hearts_alpha >= 255:
                    self.hearts_fading_in = False
            
            if self.hearts_fading_out:
                self.hearts_alpha = self._approach_alpha(self.hearts_alpha, 0, self.fade_hearts_out_duration, frame_dt)
                if self.hearts_alpha <= 0:
                    self.hearts_fading_out = False
            
            # ===== ACTUALIZAR FADE TITULOS =====
            if self.end_title_fade_in:
                self.end_title_alpha = self._approach_alpha(self.end_title_alpha, 255, self.fade_end_title_duration, frame_dt)
                if self.end_title_alpha >= 255:
                    self.end_title_fade_in = False
            
            # ===== ACTUALIZAR MENSAJES PROGRESO =====
            player_right_tile = int((self.player.pos[0] + self.player.size[0]) // self.tile_size)
            
            if (self.state == self.RUNNING) and (not show_end) and (self.msg_interval_index < len(self.msg_intervals)):
                start_t, end_t = self.msg_intervals[self.msg_interval_index]
            
                if self.msg_state == "HIDDEN" and player_right_tile >= start_t:
                    self.msg_state = "FADING_IN"
                    self.msg_alpha = 0
            
                if self.msg_state in ("FADING_IN", "SHOWN") and player_right_tile >= end_t:
                    self.msg_state = "FADING_OUT"
            
                if self.msg_state == "FADING_IN":
                    self.msg_alpha = self._approach_alpha(self.msg_alpha, 255, self.fade_msg_duration, frame_dt)
                    if self.msg_alpha >= 255:
                        self.msg_state = "SHOWN"
            
                elif self.msg_state == "FADING_OUT":
                    self.msg_alpha = self._approach_alpha(self.msg_alpha, 0, self.fade_msg_duration, frame_dt)
                    if self.msg_alpha <= 0:
                        self.msg_state = "HIDDEN"
                        self.msg_interval_index += 1
            else:
                self.msg_state = "HIDDEN"
                self.msg_alpha = 0
                       
            # ===== HUD GAMER TAG =====
            tag_height = 0
            if (self.hearts_alpha > 0) and ((not show_end) or self.hearts_fading_out):
                tag_surf = self.render_bitmap_text_surface(
                    self.gamer_tag, self.bitmap_font_chibi, spacing=self.bitmap_spacing_chibi
                )
                tag_surf.set_alpha(int(self.hearts_alpha))
                
                margin = 8
                tx, ty = margin, margin
                
                self.display.blit(tag_surf, (tx, ty))
                tag_height = tag_surf.get_height()
                
            # ===== HUD CLOCK =====
            if self.state == self.WAITING:
                progress = (self.time_limit - self.time_left) / self.time_limit
                progress = max(0.0, min(progress, 0.9999))
                clock_idx = int(progress * len(self.clock_frames))
                clock_img = self.clock_frames[clock_idx]
                
                margin = 8
                padding = 8
                cx = margin
                cy = margin + tag_height + (padding if tag_height > 0 else 0)
                
                self.display.blit(clock_img, (cx, cy))
                
            # ===== HUD VIDAS =====
            if (self.hearts_alpha > 0) and ((not show_end) or self.hearts_fading_out):
                heart_idx = self.max_lives - self.lives
                heart_idx = max(0, min(heart_idx, 3))
            
                heart_img = self.heart_small[heart_idx].copy()
                heart_img.set_alpha(int(self.hearts_alpha))
            
                margin = 8
                hx = self.base_size[0] - heart_img.get_width() - margin
                hy = margin
                self.display.blit(heart_img, (hx, hy))
                
            # ===== MENSAJES DE PROGRESO =====
            if (self.state == self.RUNNING) and (self.msg_alpha > 0) and (not show_end):
                remaining = max(0, self.total_challenges - self.stop_index)
                
                label1, val1 = "GLOBOS RESTANTES", str(remaining)
                label2, val2 = "PUNTUACION", str(self.score)
                
                s1_label = self.render_bitmap_text_surface(label1, self.bitmap_font, spacing=self.bitmap_spacing)
                s1_val   = self.render_bitmap_text_surface(val1, self.bitmap_font, spacing=self.bitmap_spacing)
                s2_label = self.render_bitmap_text_surface(label2, self.bitmap_font, spacing=self.bitmap_spacing)
                s2_val   = self.render_bitmap_text_surface(val2, self.bitmap_font, spacing=self.bitmap_spacing)
                
                for s in [s1_label, s1_val, s2_label, s2_val]:
                    s.set_alpha(int(self.msg_alpha))
                
                cx = self.base_size[0] // 2
                cy = self.base_size[1] // 4
                y_gap = 6
                
                x_label = cx - 144
                x1_val = cx + 144
                
                if self.score == 0:
                    x2_val = cx + 144
                else:
                    x2_val = cx + 128
                
                y1 = cy - s1_label.get_height() - y_gap
                y2 = cy + y_gap
                
                self.display.blit(s1_label, (x_label, y1))
                self.display.blit(s1_val,   (x1_val, y1))
                self.display.blit(s2_label, (x_label, y2))
                self.display.blit(s2_val,   (x2_val, y2))
            
            # ===== PANTALLA FINAL (WIN / GAME OVER) VERSION MINECRAFT =====
            show_end = (self.state == self.WIN) or (self.state == self.GAME_OVER and self.go_phase == "STOP")
            
            if show_end:
                completed = sum(1 for ch in self.challenges if ch.get("scored", False))
            
                # Textos normales (1x)
                end_alpha = int(self.end_title_alpha)
                
                s_gr = self.render_bitmap_text_surface("GLOBOS ROTOS", self.bitmap_font, spacing=self.bitmap_spacing)
                s_completed = self.render_bitmap_text_surface(str(completed), self.bitmap_font, spacing=self.bitmap_spacing)
                s_pts = self.render_bitmap_text_surface("PUNTUACION", self.bitmap_font, spacing=self.bitmap_spacing)
                s_score = self.render_bitmap_text_surface(str(self.score), self.bitmap_font, spacing=self.bitmap_spacing)
                
                for s in (s_gr, s_completed, s_pts, s_score):
                    s.set_alpha(end_alpha)
                
                self.display.blit(s_gr, (11*16, 11*16))
                if self.score == 0:
                    self.display.blit(s_completed, (25*16, 11*16))
                else:
                    self.display.blit(s_completed, (26*16, 11*16))
                
                self.display.blit(s_pts, (11*16, 12*16))
                self.display.blit(s_score, (25*16, 12*16))
            
                # Títulos grandes (5x)
                if self.state == self.GAME_OVER and self.go_phase == "STOP":
                    t1 = self.render_bitmap_text_surface("GAME", self.bitmap_font_big, spacing=self.bitmap_spacing_big)
                    t2 = self.render_bitmap_text_surface("OVER", self.bitmap_font_big, spacing=self.bitmap_spacing_big)
                    t1.set_alpha(int(self.end_title_alpha))
                    t2.set_alpha(int(self.end_title_alpha))
                    self.display.blit(t1, (11*16, 1*16))
                    self.display.blit(t2, (11*16, 6*16))
                
                elif self.state == self.WIN:
                    t1 = self.render_bitmap_text_surface("NIVEL", self.bitmap_font_big, spacing=self.bitmap_spacing_big)
                    t2 = self.render_bitmap_text_surface("COMPLETADO", self.bitmap_font_mid, spacing=self.bitmap_spacing_mid)
                    t1.set_alpha(int(self.end_title_alpha))
                    t2.set_alpha(int(self.end_title_alpha))
                    self.display.blit(t1, (11*16, 1*16))
                    self.display.blit(t2, (11*16, 6*16))
                
                            
            # ===== ACTUALIZAR PANTALLA Y RELOJ FPS ===== 
            self.screen.fill((0, 0, 0))
            scaled = pygame.transform.scale(self.display, self.blit_size)
            self.screen.blit(scaled, self.blit_offset)
            pygame.display.update()

            self.clock.tick(60)
    
    def run_level_1(self):
        self.reset_level(1)
        self._run_level_shared()

    def run_level_2(self):
        self.reset_level(2)
        self._run_level_shared()
    
    def reset_keys(self):
        self.UP_KEY = self.DOWN_KEY = self.START_KEY = self.BACK_KEY = False
        self.TEXT_INPUT = ""
        self.BACKSPACE = False

    def check_events(self):
        self.UP_KEY = self.DOWN_KEY = self.START_KEY = self.BACK_KEY = False
    
        self.TEXT_INPUT = ""
        self.BACKSPACE = False
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.playing = False
                pygame.quit()
                return
    
            if event.type == pygame.TEXTINPUT:
                self.TEXT_INPUT += event.text
    
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                elif event.key == pygame.K_UP:
                    self.UP_KEY = True
                elif event.key == pygame.K_DOWN:
                    self.DOWN_KEY = True
                elif event.key == pygame.K_RETURN:
                    self.START_KEY = True
                elif event.key == pygame.K_ESCAPE:
                    self.BACK_KEY = True
    
                elif event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
                    self.BACKSPACE = True


# ===== EJECUTABLE =====

if __name__ == "__main__":
    Game().run()