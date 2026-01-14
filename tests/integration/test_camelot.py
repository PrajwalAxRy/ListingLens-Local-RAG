import camelot
import os

path = r"C:\Projects\listing-lens-RAG-Test\input\RHP_Modified.pdf"
script_dir = os.path.dirname(os.path.abspath(__file__))

# Target page 8 lattice for bordered table
tables = camelot.read_pdf(path, pages="8", flavor="lattice")
print(f"Found {len(tables)} tables with lattice on page 8")

if tables:
    print("Table head:")
    print(tables[0].df.head())
    print("\nParsing report:")
    print(tables[0].parsing_report)
    
    # Save with descriptive name in same directory
    csv_path = os.path.join(script_dir, "rhp_employee_benefits_table.csv")
    xlsx_path = os.path.join(script_dir, "rhp_employee_benefits_table.xlsx")
    
    tables[0].to_csv(csv_path)
    tables[0].to_excel(xlsx_path)
    
    print(f"\nSaved:")
    print(f"  CSV:  {csv_path}")
    print(f"  Excel: {xlsx_path}")
    print(f"Accuracy: {tables[0].parsing_report['accuracy']:.1%}")
else:
    print("Lattice failed—check PDF rendering or try stream fallback")
