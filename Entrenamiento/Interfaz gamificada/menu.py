import pygame

class Menu:
    def __init__(self, game):
        self.game = game
        self.run_display = True
        self.state = None
        
        # ===== CONSTANTE DE DESPLAZAMIENTO VERTICAL (SIN ESTO NO CUADRAN LOS PX) =====
        self.offset_y = 5

    def draw_cursor(self, option_rect):
        cursor = self.game.render_bitmap_text_surface(
            ">", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
        )
        cursor_rect = cursor.get_rect(midright=(option_rect.left - 10, option_rect.centery))
        self.game.display.blit(cursor, cursor_rect)

    def blit_screen(self):
        self.game.screen.fill((0, 0, 0))
        scaled = pygame.transform.scale(self.game.display, self.game.blit_size)
        self.game.screen.blit(scaled, self.game.blit_offset)
        pygame.display.update()
        self.game.reset_keys()
        self.game.clock.tick(60)
        
        
# ===== MENU PRINCIPAL =====
class MainMenu(Menu):
    def __init__(self, game):
        super().__init__(game)
        self.options = ["Niveles", "Configuracion"]
        self.index = 0

    def display_menu(self):
        self.run_display = True
        while self.run_display:
            self.game.check_events()
            self.check_input()

            self.game.display.blit(self.game.menu_bg, (0, 0))
            mid_w, _ = self.game.display.get_rect().center

            # ===== TITULO DE MAIN MENU =====
            title = self.game.render_bitmap_text_surface(
                "MAIN MENU", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
            )
            title_rect = title.get_rect(center=(mid_w, 50 + self.offset_y))
            self.game.display.blit(title, title_rect)

            # ===== ALTURAS DE NIVELES Y CONFIG =====
            posiciones_y = [140, 170]
            rects = []
            for i, text in enumerate(self.options):
                surf = self.game.render_bitmap_text_surface(
                    text, self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
                )
                rect = surf.get_rect(center=(mid_w, posiciones_y[i] + self.offset_y))
                self.game.display.blit(surf, rect)
                rects.append(rect)

            self.draw_cursor(rects[self.index])
            self.blit_screen()

    def check_input(self):
        if self.game.DOWN_KEY:
            self.index = (self.index + 1) % len(self.options)
        if self.game.UP_KEY:
            self.index = (self.index - 1) % len(self.options)

        if self.game.START_KEY:
            choice = self.options[self.index]

            if choice == "Niveles":
                self.game.curr_menu = self.game.levels_menu
                self.run_display = False

            elif choice == "Configuracion":
                self.game.curr_menu = self.game.config_menu
                self.run_display = False


# ===== VENTANA MENU DE CONFIG =====
class ConfigMenu(Menu):
    def __init__(self, game):
        super().__init__(game)
        self.state = "Volver"          # cursor puede estar en "Volver" o "BT"
        
        self.nickname_buffer = ""
        self.max_nickname_len = 20
        self.cursor_blink_ms = 500

    def display_menu(self):
        self.run_display = True
        pygame.key.start_text_input()
        pygame.key.set_repeat(300, 35)
        self.nickname_buffer = self.game.gamer_tag

        while self.run_display:
            self.game.check_events()

            # ── Nickname input ──────────────────────────────
            if self.game.TEXT_INPUT:
                self.nickname_buffer += self.game.TEXT_INPUT
            if self.game.BACKSPACE and len(self.nickname_buffer) > 0:
                self.nickname_buffer = self.nickname_buffer[:-1]
            allowed = set(self.game.bitmap_charset)
            self.nickname_buffer = "".join(
                ch for ch in self.nickname_buffer if ch in allowed
            )
            self.nickname_buffer = self.nickname_buffer[:self.max_nickname_len]
            self.check_input()

            self.game.display.blit(self.game.menu_bg, (0, 0))
            mid_w, _ = self.game.display.get_rect().center

            # ── Título ──────────────────────────────────────
            title = self.game.render_bitmap_text_surface(
                "CONFIGURACION", self.game.bitmap_font_white,
                spacing=self.game.bitmap_spacing
            )
            self.game.display.blit(
                title, title.get_rect(center=(mid_w, 50 + self.offset_y))
            )

            # ── Nickname label ──────────────────────────────
            nick_label = self.game.render_bitmap_text_surface(
                "Nickname", self.game.bitmap_font_white,
                spacing=self.game.bitmap_spacing
            )
            self.game.display.blit(
                nick_label,
                nick_label.get_rect(center=(mid_w, 110 + self.offset_y))
            )

            # ── Nickname typing + caret ─────────────────────
            typed_y   = 140 + self.offset_y
            base_text = self.nickname_buffer

            if base_text:
                base_surf = self.game.render_bitmap_text_surface(
                    base_text, self.game.bitmap_font_white,
                    spacing=self.game.bitmap_spacing
                )
                base_rect = base_surf.get_rect(center=(mid_w, typed_y))
                self.game.display.blit(base_surf, base_rect)
            else:
                base_rect = None

            if (pygame.time.get_ticks() // self.cursor_blink_ms) % 2 == 0:
                caret_surf = self.game.render_bitmap_text_surface(
                    "|", self.game.bitmap_font_white,
                    spacing=self.game.bitmap_spacing
                )
                if base_rect:
                    caret_rect = caret_surf.get_rect(
                        midleft=(base_rect.right + 2, base_rect.centery)
                    )
                else:
                    caret_rect = caret_surf.get_rect(center=(mid_w, typed_y))
                self.game.display.blit(caret_surf, caret_rect)

            # ── Switch Bluetooth ────────────────────────────
            bt_y = 180 + self.offset_y
            self._draw_bt_switch(mid_w, bt_y)

            # ── Volver ──────────────────────────────────────
            volver = self.game.render_bitmap_text_surface(
                "Volver", self.game.bitmap_font_white,
                spacing=self.game.bitmap_spacing
            )
            volver_rect = volver.get_rect(center=(mid_w, 220 + self.offset_y))
            self.game.display.blit(volver, volver_rect)
            

            # Cursor solo en "Volver" (el BT se activa con ENTER sobre él)
            if self.state == "Volver":
                self.draw_cursor(volver_rect)
            if self.state == "BT":
                self.draw_cursor(self._bt_rect.move(-90, 0))  # ajusta según el diseño
                  
            self.blit_screen()

        pygame.key.stop_text_input()
        pygame.key.set_repeat()

    # ── Dibuja el switch ────────────────────────────────────
    def _draw_bt_switch(self, mid_w, y):
        """
        Dibuja:  Bluetooth  [OFF]   o   Bluetooth  [ON ]
        El rectángulo cambia de color según el estado.
        """
        # Label izquierdo
        label = self.game.render_bitmap_text_surface(
            "ARIADNE", self.game.bitmap_font_white,
            spacing=self.game.bitmap_spacing
        )
        label_rect = label.get_rect(midright=(mid_w +10, y))
        self.game.display.blit(label, label_rect)

        # Caja del switch
        enabled = self.game.bluetooth_enabled
        box_color  = (0, 200, 80)  if enabled else (0, 0, 0)
        text_str   = "ON " if enabled else "OFF"

        box_w, box_h = 36, 14          # ajusta a la escala de tu display
        box_rect = pygame.Rect(0, 0, box_w, box_h)
        box_rect.midleft = (mid_w + 20, y)
        pygame.draw.rect(self.game.display, box_color, box_rect, border_radius=3)

        switch_text = self.game.render_bitmap_text_surface(
            text_str, self.game.bitmap_font_white,
            spacing=self.game.bitmap_spacing
        )
        self.game.display.blit(
            switch_text,
            switch_text.get_rect(center=box_rect.center)
        )

        # Guardamos el rect para check_input (detección de cursor opcional)
        self._bt_rect = box_rect

    def check_input(self):
        if self.game.DOWN_KEY:
            # Alternar entre "BT" y "Volver"
            self.state = "Volver" if self.state == "BT" else "BT"
        if self.game.UP_KEY:
            self.state = "Volver" if self.state == "BT" else "BT"

        if self.game.START_KEY:
            if self.state == "BT":
                # Toggle
                self.game.bluetooth_enabled = not self.game.bluetooth_enabled
            elif self.state == "Volver":
                self._save_and_return()

        if self.game.BACK_KEY:
            self._save_and_return()

    def _save_and_return(self):
        cleaned = self.nickname_buffer.strip()
        if cleaned:
            self.game.gamer_tag = cleaned
        self.game.curr_menu = self.game.main_menu
        self.run_display = False


# ===== VENTANA NIVELES =====
class LevelsMenu(Menu):
    def __init__(self, game):
        super().__init__(game)
        self.options = ["Nivel 1", "Nivel 2", "Volver"]
        self.index = 0

    def display_menu(self):
        self.run_display = True
        while self.run_display:
            self.game.check_events()
            self.check_input()

            self.game.display.blit(self.game.menu_bg, (0, 0))
            mid_w, _ = self.game.display.get_rect().center

            # ===== TITULO DE NIVELES =====
            title = self.game.render_bitmap_text_surface(
                "NIVELES", self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
            )
            title_rect = title.get_rect(center=(mid_w, 50 + self.offset_y))
            self.game.display.blit(title, title_rect)

            # ===== ALTURAS PARA LVL 1, LVL 2 Y VOLVER =====
            posiciones_y = [110, 140, 200]
            option_rects = []
            for i, opt in enumerate(self.options):
                surf = self.game.render_bitmap_text_surface(
                    opt, self.game.bitmap_font_white, spacing=self.game.bitmap_spacing
                )
                rect = surf.get_rect(center=(mid_w, posiciones_y[i] + self.offset_y))
                self.game.display.blit(surf, rect)
                option_rects.append(rect)

            self.draw_cursor(option_rects[self.index])
            self.blit_screen()

    def check_input(self):
        if self.game.DOWN_KEY:
            self.index = (self.index + 1) % len(self.options)
        if self.game.UP_KEY:
            self.index = (self.index - 1) % len(self.options)

        if self.game.BACK_KEY:
            self.game.curr_menu = self.game.main_menu
            self.run_display = False

        if self.game.START_KEY:
            choice = self.options[self.index]

            if choice == "Volver":
                self.game.curr_menu = self.game.main_menu
                self.run_display = False

            elif choice == "Nivel 1":
                self.game.selected_level = 1
                self.game.playing = True
                self.run_display = False

            elif choice == "Nivel 2":
                self.game.selected_level = 2
                self.game.playing = True
                self.run_display = False
