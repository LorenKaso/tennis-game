import pandas as pd                             # לקריאת קובץ CSV
import numpy as np                              # לעבודה עם מטריצות
import cv2
import matplotlib.pyplot as plt                 # לציור הגרפים
from matplotlib.colors import LinearSegmentedColormap  # להגדרת צבעים מותאמים אישית
from scipy.ndimage import gaussian_filter       # לסינון מטושטש של ההיטמאפ
from matplotlib.colors import ListedColormap

def plot_with_background(data, title, save_path, background_path):
    background = cv2.imread(background_path)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    
    h, w = background.shape[:2]
    plt.figure(figsize=(12, 7))
    plt.imshow(background)
    
    # שמירת פרופורציות וגבולות
    plt.gca().set_aspect('auto')
    plt.xlim(0, w)
    plt.ylim(h, 0)

    # colormap 
    base = plt.cm.turbo(np.linspace(0, 1, 256))
    base[:40, -1] = 0.0     
    base[40:100, -1] = 0.3  
    base[100:200, -1] = 0.6 
    base[200:, -1] = 0.9    

    cmap = ListedColormap(base)
    data = gaussian_filter(data, sigma=50)
    data = data / np.max(data)
    plt.imshow(data, cmap=cmap, alpha=1.0, interpolation='bilinear')

    im = plt.imshow(data, cmap=cmap, alpha=1.0) 
    cbar = plt.colorbar(im, label="Density")     
    cbar.set_ticks(np.arange(0, 11, 1))           # מספרים שלמים בלבד
    im.set_clim(0, 10.5)                            # תחום המדד לתמונה

    plt.title(title)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

if __name__ == "__main__":
    w = 1280                           # רוחב של ההיטמאפ (כמו הווידאו)
    h = 720                            # גובה של ההיטמאפ (כמו הווידאו)

    # טוען את קובץ מיקומי השחקנים שנוצר קודם
    df = pd.read_csv("player_positions.csv")

    # מאחד את כל מיקומי השחקנים (שמאלי וימני) לרשימה אחת של x ו-y
    x = list(df["player1_position_x"]) + list(df["player2_position_x"])
    y = list(df["player1_position_y"]) + list(df["player2_position_y"])

    # יוצר מטריצה ריקה בגודל הסרטון – כל תא יצבור "כמה פעמים היו פה"
    data = np.zeros((h, w))

    # ממפה כל נקודת מיקום לתוך התא המתאים במטריצה
    for i in range(len(x)):
        xi = int(min(max(x[i], 0), w - 1))  # מבטיח שה-x בטווח [0, w)
        yi = int(min(max(y[i], 0), h - 1))  # מבטיח שה-y בטווח [0, h)
        data[yi][xi] += 1                   # מגדיל את הספירה בתא המתאים

    # החלקה של המטריצה עם פילטר גאוסי כדי לקבל אפקט מטושטש
    data = gaussian_filter(data, sigma=20)

    # ציור המפה וכתיבתה לקובץ PNG
    plot_with_background(
    data,
    "Player Position Heatmap on Court",
    "player_position_heatmap.png",
    "background_frame.png"
    )

