import pygame
import time

class UIComponent:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.is_hovered = False
        self.is_pressed = False

    def update(self, mouse_pos, mouse_down):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        self.is_pressed = self.is_hovered and mouse_down

class Button(UIComponent):
    def __init__(self, x, y, w, h, text, color=(52, 152, 219), font_size=20):
        super().__init__(x, y, w, h)
        self.text = text
        self.base_color = color
        self.font = pygame.font.SysFont("Arial", font_size, bold=True)
        self.surface = self.font.render(self.text, True, (255, 255, 255))
        self.text_rect = self.surface.get_rect(center=self.rect.center)

    def draw(self, screen):
        # Background
        color = self.base_color
        if self.is_pressed:
            color = tuple(max(0, c - 40) for c in self.base_color)
        elif self.is_hovered:
            color = tuple(min(255, c + 30) for c in self.base_color)
        
        # Draw with slight shadow/glow
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, width=1, border_radius=10)
        
        # Text
        screen.blit(self.surface, self.text_rect)

class Slider(UIComponent):
    def __init__(self, x, y, w, h, min_val, max_val, initial_val, label="", is_int=True):
        super().__init__(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.is_int = is_int
        self.font = pygame.font.SysFont("Arial", 16)
        
        # Interaction rect (the handle)
        self.handle_width = 15
        self.handle_rect = pygame.Rect(0, y - 5, self.handle_width, h + 10)
        self.update_handle_pos()
        self.dragging = False

    def update_handle_pos(self):
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        self.handle_rect.centerx = self.rect.x + ratio * self.rect.w

    def update(self, mouse_pos, mouse_down):
        super().update(mouse_pos, mouse_down)
        
        if self.is_hovered and mouse_down and not self.dragging:
            self.dragging = True
            
        if not mouse_down:
            self.dragging = False
            
        if self.dragging:
            rel_x = max(0, min(self.rect.w, mouse_pos[0] - self.rect.x))
            ratio = rel_x / self.rect.w
            self.value = self.min_val + ratio * (self.max_val - self.min_val)
            if self.is_int:
                self.value = int(round(self.value))
            self.update_handle_pos()

    def draw(self, screen):
        # Label and Value
        val_text = f"{self.value}" if self.is_int else f"{self.value:.2f}"
        text_surf = self.font.render(f"{self.label}: {val_text}", True, (200, 200, 200))
        screen.blit(text_surf, (self.rect.x, self.rect.y - 25))
        
        # Track
        pygame.draw.rect(screen, (40, 40, 60), self.rect, border_radius=3)
        
        # Handle
        color = (52, 152, 219) if self.dragging or self.is_hovered else (41, 128, 185)
        pygame.draw.rect(screen, color, self.handle_rect, border_radius=5)
        pygame.draw.rect(screen, (255, 255, 255), self.handle_rect, width=1, border_radius=5)

class SidebarItem(UIComponent):
    def __init__(self, x, y, w, h, label, icon_char=""):
        super().__init__(x, y, w, h)
        self.label = label
        self.icon_char = icon_char
        self.font = pygame.font.SysFont("Arial", 18)
        self.selected = False

    def draw(self, screen):
        bg_color = (15, 15, 25)
        text_color = (150, 150, 150)
        
        if self.selected:
            bg_color = (30, 40, 60)
            text_color = (255, 255, 255)
            # Selection indicator
            pygame.draw.rect(screen, (52, 152, 219), (self.rect.x, self.rect.y, 4, self.rect.h))
        elif self.is_hovered:
            bg_color = (25, 25, 35)
            text_color = (200, 200, 200)
            
        pygame.draw.rect(screen, bg_color, self.rect)
        
        txt = self.font.render(self.label, True, text_color)
        screen.blit(txt, (self.rect.x + 20, self.rect.y + (self.rect.h - txt.get_height()) // 2))

# --- Colors ---
COLORS = {
    "bg": (10, 10, 15),
    "sidebar": (15, 15, 25),
    "panel": (20, 20, 30),
    "text": (240, 240, 245),
    "accent": (52, 152, 219),
    "secondary": (46, 204, 113)
}
