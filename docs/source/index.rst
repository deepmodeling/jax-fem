.. jax-fem-docs documentation master file, created by
   sphinx-quickstart on Thu Jul  3 15:46:45 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to JAX-FEM's documentation!
===================================

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.

🚀 `JAX-FEM <https://github.com/deepmodeling/jax-fem>`_ is a differentiable finite element package based on `JAX <https://github.com/google/jax>`_.

.. raw:: html

   <div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
   <figure style="margin: 0; text-align: center;">
      <img src="_static/images/von_mises.png" style="height: 200px; width: auto; object-fit: contain;" />
      <p style="margin-top: 10px;"><em>Linear static analysis of a bracket.</em></p>
   </figure>
   <figure style="margin: 0; text-align: center;">
      <img src="_static/images/to.gif" style="height: 200px; width: auto; object-fit: contain;" />
      <p style="margin-top: 10px;"><em>Topology optimization with differentiable simulation.</em></p>
   </figure>
   </div>

   <div style="display: flex; justify-content: center; align-items: flex-start; gap: 60px;">
      <figure style="margin: 0; text-align: center;">
         <img src="_static/images/stokes_u.png" style="height: 200px; width: auto; object-fit: contain;" />
      </figure>
      <figure style="margin: 0; text-align: center;">
         <img src="_static/images/stokes_p.png" style="height: 200px; width: auto; object-fit: contain;" />
      </figure>
   </div>
   <p style="text-align: center; margin-top: 10px;"><em>Stokes flow: velocity (left) and pressure(right).</em></p>


   <div style="display: flex; justify-content: center; align-items: flex-start; gap: 60px;">
      <figure style="margin: 0; text-align: center;">
         <img src="_static/images/polycrystal_grain.gif" style="height: 260px; width: auto; object-fit: contain;" />
      </figure>
      <figure style="margin: 0; text-align: center;">
         <img src="_static/images/polycrystal_stress.gif" style="height: 260px; width: auto; object-fit: contain;" />
      </figure>
   </div>
   <p style="text-align: center; margin-top: 10px;"><em>Crystal plasticity: grain structure (left) and stress-xx (right).</em></p>


   <div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
      <figure style="margin: 0; text-align: center;">
         <img src="_static/images/ded.gif" style="height: 280px; width: auto; object-fit: contain;" />
      </figure>
   </div>
   <p style="text-align: center; margin-top: 10px;"><em>Thermal profile in direct energy deposition.</em></p>

.. toctree::
   :maxdepth: 2 
   :caption: User guides
   :hidden:

   Installation <guide/Installation>
   Quickstart <guide/Quickstart>

.. toctree::
   :maxdepth: 2
   :caption: Learn by examples
   :hidden:

   Overview <learn/overview>

   Poisson equation <learn/poisson/example>

   Linear elasticity <learn/linear_elasticity/example>

   Hyperelasticity <learn/hyperelasticity/example>

   Plasticity <learn/plasticity/example>

   Compute gradients <learn/compute_gradients/example>

   Topology optimization <learn/topology_optimization/example>

   Source field identification <learn/source_field_identification/example>

   Traction force identification <learn/traction_force_identification/example>

   Thermal mechanical control <learn/thermal_mechanical_control/example>

   Shape optimization <learn/shape_optimization/example>

   Material/structure co-design <learn/material_structure_co_design/example>

.. toctree::
   :maxdepth: 1
   :caption: Advanced topics
   :hidden:

   advanced/adv_main.md
   

.. toctree::
   :maxdepth: 3
   :caption: More resources
   :hidden:

   Frequently asked questions (FAQ) <more/FAQ>

   Useful links <more/useful/main>

   API reference <more/api/api_main>

   Change log <more/log>

   Citations <more/citation>

   Miscellaneous <more/project>

   Contact <more/contact>
   