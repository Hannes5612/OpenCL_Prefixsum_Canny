#pragma once

#include <SDL/SDL.h>
#include <SDL/SDL_ttf.h>
#include <SDL/SDL_active.h>
#include <SDL/SDL_rotozoom.h>
#include <SDL/SDL_image.h>
#include <string>
#include <sstream>

TTF_Font *font = 0;

struct Control {
    SDL_Surface *surf;
    SDL_Rect area;

    Control(SDL_Surface *surf) : surf(surf) {
        setRegion(0, 0, 0, 0);
    }

    void setRegion(int x, int y, int w, int h) {
        area.x = x; area.y = y; area.w = w; area.h = h;
    }

    virtual void draw() = 0;
    virtual void clicked(int x, int y) { }
	virtual void wheel(int dy) { }
};

struct Button : Control {
    int color;
    int textcolor;
    SDL_Surface *fontsurf;

    Button(SDL_Surface *surf) : Control(surf) {
        fontsurf = 0;
        color = 0x00808080;  // gray
        textcolor = 0;       // black
    }

    void setText(std::string text) {
        if (fontsurf) SDL_FreeSurface(fontsurf);
        fontsurf = 0;
        if (text.empty()) return;
        SDL_Color col; col.r=col.g=col.b=255;
        fontsurf = TTF_RenderText_Blended(
                    font, text.c_str(), *((SDL_Color*)&textcolor));
    }

    ~Button() {
        setText("");
    }

    virtual void draw() {
        SDL_FillRect(surf, &area, color/2);
        SDL_Rect area2 = area;
        area2.x += 2; area2.y += 2;
        area2.w -= 4; area2.h -= 4;
        SDL_FillRect(surf, &area2, color);

        if (!fontsurf) return; // no text to draw
        SDL_Rect dest;
        dest.x = area.x + (area.w - fontsurf->w)/2;
        dest.y = area.y + (area.h - fontsurf->h)/2;
        dest.w = surf->w;
        dest.h = surf->h;
        SDL_BlitSurface(fontsurf, NULL, surf, &dest);
    }
};

struct Image : Control {
    SDL_Surface *imgsurf;
    Image(SDL_Surface *surf) : Control(surf) {
        imgsurf = 0;
        SDL_Color col; col.r=col.g=col.b=64;
        setContent(TTF_RenderText_Blended(font, "x", col));
    }
    void setContent(SDL_Surface *x) {
        if (imgsurf) SDL_FreeSurface(imgsurf);
        imgsurf = x;
    }
    virtual void draw() {
        SDL_FillRect(surf, &area, 0);
        if (!imgsurf) return;
        float scale = 1;
        scale = float(area.w)/imgsurf->w;
        float scale2 = float(area.h)/(imgsurf->h*scale);
        if (scale2 < 1) scale *= scale2;
        SDL_Surface *rz = zoomSurface(imgsurf, scale, scale, 1);
        SDL_Rect dest;
        dest.x = area.x + area.w/2 - rz->w/2;
        dest.y = area.y + area.h/2 - rz->h/2;
        dest.w = rz->w;
        dest.h = rz->h;
        SDL_BlitSurface(rz, NULL, surf, &dest);
        SDL_FreeSurface(rz);
    }
};

#include <vector>

struct Window {
    SDL_Surface *screen, *pic;
    std::vector<Control*> controls;
    int grid;
    //Value *vs[8];
    Button *apply, *reset, *save;
    Image *image;
    SDL_TimerID timer;
	Control *clicked;
	int timerevents;

    Window() {
		timer = 0;
		screen = 0; pic = 0;
		grid = 40;
        if (SDL_Init(SDL_INIT_VIDEO|SDL_INIT_TIMER)) throw "SDL init";
        if (TTF_Init()) throw "TTF init";
        if (!IMG_Init(~0)) throw "IMG init";
        font = TTF_OpenFont("LiberationSans-Bold.ttf", 24);
        if (!font) throw "open font";
        screen = SDL_SetVideoMode(1920, 1360, 32, SDL_HWSURFACE|SDL_RESIZABLE);
        if (!screen) throw "create screen";

		apply = new Button(screen);
        apply->setText("apply");
        controls.push_back(apply);

        reset = new Button(screen);
        reset->setText("reset");
        controls.push_back(reset);

		save = new Button(screen);
		save->setText("save BMP");
		controls.push_back(save);

        image = new Image(screen);
        controls.push_back(image);

        arrange();
		clicked = 0;
		timerevents = 0;
    }

    void useBuffer(int32_t *buf, int w, int h) {
        SDL_Surface *s = SDL_CreateRGBSurfaceFrom(
                    buf, w, h, 32, w*4, 0x00FF0000, 0x0000FF00, 0x000000FF, 0);
        image->setContent(s);
    }

    void arrange() {
        int w = screen->w, h = screen->h;
        //for (int i=0; i<8; i++) {
        //    vs[i]->setRegion(w-grid*4, i*grid, grid*4, grid);
        //}
		apply->setRegion(w - grid * 4, h - grid * 3, grid * 4, grid);
		reset->setRegion(w - grid * 4, h - grid * 2, grid * 4, grid);
		save->setRegion(w - grid * 4, h - grid, grid * 4, grid);
		image->setRegion(0, 0, w-grid*4, h);
    }

    void draw() {
        for (int i=0; i<controls.size(); i++)
            controls[i]->draw();
    }

    static Uint32 timer_callbackfunc(Uint32 interval, void *_this)
    {
        SDL_Event event;
        SDL_UserEvent userevent;

        userevent.type = SDL_USEREVENT;
        userevent.code = 0;
        userevent.data1 = NULL;
        userevent.data2 = NULL;

        event.type = SDL_USEREVENT;
        event.user = userevent;

		int &timerevents = ((Window*)_this)->timerevents;
		if (!timerevents) {
	        SDL_PushEvent(&event);
			timerevents++;
		}

        return(interval);
    }

	virtual void onSave() {}
    virtual void onApply() {}
    virtual void onReset() {}

	void saveBMP(std::string fname) {
		SDL_SaveBMP(image->imgsurf, fname.c_str());
	}

    void run() {
        SDL_Event event;
        draw();
        SDL_Flip(screen);
        while (SDL_WaitEvent(&event)) {
            if (event.type == SDL_QUIT) return;
            if (event.type == SDL_USEREVENT) {
				// event is generated by timer
				timerevents--;
                onApply();
                draw();
                SDL_Flip(screen);
				SDL_GetTicks();
            }
            if (event.type == SDL_MOUSEBUTTONDOWN) {
                for (int i=0; i<controls.size(); i++) {
                    int dx = event.button.x - controls[i]->area.x;
                    int dy = event.button.y - controls[i]->area.y;
                    if (dx >= 0 && dx < controls[i]->area.w &&
                        dy >= 0 && dy < controls[i]->area.h) {
						int index = event.button.button;
						if (index == 4 || index == 5) {
							// mouse wheel
							if (clicked) {
								clicked->wheel((index==4) ? 1 : -1);
								draw();
								SDL_Flip(screen);								
							}
							continue;
						}

						// normal click
                        if (controls[i] == apply) 
							onApply();
						else if (controls[i] == reset)
							onReset();
						else if (controls[i] == save)
							onSave();
                        
                        controls[i]->clicked(dx, dy);
                        draw();
                        SDL_Flip(screen);
						clicked = controls[i];  // remember for wheel events
                    }
                }
			}
            if (event.type == SDL_VIDEORESIZE) {
                SDL_FreeSurface(screen);
                screen = SDL_SetVideoMode(event.resize.w, event.resize.h, 32, SDL_SWSURFACE|SDL_RESIZABLE);
                for (int i=0; i<controls.size(); i++)
                    controls[i]->surf = screen;
                arrange();
                draw();
                SDL_Flip(screen);
            }
			if (event.type == SDL_KEYDOWN) {
				if (event.key.keysym.sym == SDLK_F4 && event.key.keysym.mod & KMOD_LALT) {
					return;   // close when catching Alt+F4
				}
			}
        }
        throw "wait for event";
	}

    ~Window() {
        SDL_FreeSurface(pic);
        SDL_Quit();
    }
};

SDL_Surface *loadImageRGB32(std::string name) {
    SDL_Surface *surf = IMG_Load(name.c_str());
	if (!surf) throw "load image";
    SDL_PixelFormat fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.Aloss = 8;
    fmt.BitsPerPixel = 32;
    fmt.BytesPerPixel = 4;
    fmt.Rmask = 0x00FF0000; fmt.Rshift = 24;
    fmt.Gmask = 0x0000FF00; fmt.Gshift = 16;
    fmt.Bmask = 0x000000FF; fmt.Bshift = 8;
    SDL_Surface *out = SDL_ConvertSurface(surf, &fmt, SDL_SWSURFACE);
    SDL_FreeSurface(surf);
    return out;
}
